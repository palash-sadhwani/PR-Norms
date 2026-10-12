"""Deterministic semantic embedding, clustering, and candidate norm mining."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import (
    CandidateNorm,
    CandidateNormEvidence,
    CommentCluster,
    CommentClusterMember,
    CommentEmbedding,
    PullRequest,
    Repository,
    ReviewComment,
)


class Embedder(Protocol):
    model_name: str

    def encode(
        self, texts: Sequence[str], batch_size: int
    ) -> Sequence[Sequence[float]]: ...


class Clusterer(Protocol):
    def fit_predict(self, vectors: Sequence[Sequence[float]]) -> Sequence[int]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Phase 2 requires sentence-transformers. "
                'Install with `pip install -e ".[phase2]"`.'
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(
        self, texts: Sequence[str], batch_size: int
    ) -> Sequence[Sequence[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


class HdbscanClusterer:
    def __init__(
        self, min_cluster_size: int, min_samples: int | None, seed: int
    ) -> None:
        try:
            import hdbscan
        except ImportError as exc:
            raise RuntimeError(
                'Phase 2 requires hdbscan. Install with `pip install -e ".[phase2]"`.'
            ) from exc
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
        )

    def fit_predict(self, vectors: Sequence[Sequence[float]]) -> Sequence[int]:
        return self._clusterer.fit_predict(vectors).tolist()


@dataclass(frozen=True)
class MiningResult:
    embedded: int
    clustered: int
    clusters: int
    norms: int
    eligible: int = 0
    excluded: int = 0
    noise: int = 0
    embeddings_reused: int = 0
    embeddings_created: int = 0


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    dimensions = len(vectors[0])
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimensions)
    ]


async def mine_repository(
    db: AsyncSession,
    repository_id: int,
    *,
    embedder: Embedder | None = None,
    clusterer: Clusterer | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> MiningResult:
    settings = get_settings()
    model_name = model_name or settings.phase2_embedding_model
    batch_size = batch_size or settings.phase2_embedding_batch_size
    repository = await db.get(Repository, repository_id)
    if repository is None:
        raise ValueError(f"repository {repository_id} not found")

    comments = list(
        (
            await db.scalars(
                select(ReviewComment)
                .join(PullRequest, PullRequest.id == ReviewComment.pull_request_id)
                .where(
                    PullRequest.repository_id == repository_id,
                    ReviewComment.is_filtered.is_(False),
                    ReviewComment.normalized_body.is_not(None),
                )
                .order_by(ReviewComment.id)
            )
        ).all()
    )
    comments = [comment for comment in comments if comment.normalized_body]
    total_comments = await db.scalar(
        select(func.count(ReviewComment.id))
        .join(PullRequest, PullRequest.id == ReviewComment.pull_request_id)
        .where(PullRequest.repository_id == repository_id)
    )
    excluded = int(total_comments or 0) - len(comments)
    if not comments:
        return MiningResult(0, 0, 0, 0, eligible=0, excluded=excluded)

    embedder = embedder or SentenceTransformerEmbedder(model_name)
    existing_embeddings = {
        embedding.review_comment_id: embedding
        for embedding in (
            await db.scalars(
                select(CommentEmbedding).where(
                    CommentEmbedding.review_comment_id.in_(
                        [comment.id for comment in comments]
                    ),
                    CommentEmbedding.model_name == model_name,
                )
            )
        ).all()
    }
    hashes = {
        comment.id: _hash_text(comment.normalized_body or "") for comment in comments
    }
    vectors_by_comment: dict[int, Sequence[float]] = {}
    missing_comments = []
    reused = 0
    for comment in comments:
        embedding = existing_embeddings.get(comment.id)
        if (
            embedding is not None
            and embedding.source_text_hash == hashes[comment.id]
            and embedding.vector
        ):
            vectors_by_comment[comment.id] = embedding.vector
            reused += 1
        else:
            missing_comments.append(comment)
    if missing_comments:
        generated = embedder.encode(
            [comment.normalized_body for comment in missing_comments],
            batch_size=batch_size,
        )
        for comment, vector in zip(missing_comments, generated):
            vectors_by_comment[comment.id] = vector
    vectors = [vectors_by_comment[comment.id] for comment in comments]
    if len(vectors) != len(comments):
        raise ValueError("embedding provider returned an unexpected vector count")
    dimensions = len(vectors[0])
    if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("embedding provider returned inconsistent vector dimensions")

    for comment, vector in zip(comments, vectors):
        embedding = existing_embeddings.get(comment.id)
        if embedding is None:
            embedding = CommentEmbedding(
                review_comment_id=comment.id,
                model_name=model_name,
            )
            db.add(embedding)
        embedding.source_text_hash = _hash_text(comment.normalized_body or "")
        embedding.dimensions = len(vector)
        embedding.vector = [float(value) for value in vector]

    if clusterer is None and len(vectors) < settings.phase2_min_cluster_size:
        await db.commit()
        return MiningResult(
            len(comments),
            0,
            0,
            0,
            eligible=len(comments),
            excluded=excluded,
            embeddings_reused=reused,
            embeddings_created=len(missing_comments),
        )

    clusterer = clusterer or HdbscanClusterer(
        settings.phase2_min_cluster_size,
        settings.phase2_min_samples,
        settings.phase2_random_seed,
    )
    labels = list(clusterer.fit_predict(vectors))
    if len(labels) != len(comments):
        raise ValueError("clusterer returned an unexpected label count")

    old_clusters = list(
        (
            await db.scalars(
                select(CommentCluster).where(
                    CommentCluster.repository_id == repository_id,
                    CommentCluster.model_name == model_name,
                    CommentCluster.algorithm == "hdbscan",
                )
            )
        ).all()
    )
    existing_by_label = {cluster.label: cluster for cluster in old_clusters}
    for cluster in old_clusters:
        await db.execute(
            delete(CommentClusterMember).where(
                CommentClusterMember.cluster_id == cluster.id
            )
        )
    await db.flush()

    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label >= 0:
            grouped.setdefault(int(label), []).append(index)

    norm_count = 0
    noise = labels.count(-1)
    for label, indices in sorted(grouped.items()):
        group_vectors = [vectors[index] for index in indices]
        center = _centroid(group_vectors)
        similarities = [cosine_similarity(vectors[index], center) for index in indices]
        confidence = min(
            1.0,
            (len(indices) / max(len(comments), 1)) * 0.5
            + (sum(similarities) / len(similarities)) * 0.5,
        )
        cluster = existing_by_label.get(label)
        if cluster is None:
            cluster = CommentCluster(
                repository_id=repository_id,
                model_name=model_name,
                algorithm="hdbscan",
                algorithm_config={},
                label=label,
                size=0,
                confidence=0.0,
            )
            db.add(cluster)
        cluster.algorithm_config = {
            "min_cluster_size": settings.phase2_min_cluster_size,
            "min_samples": settings.phase2_min_samples,
            "random_seed": settings.phase2_random_seed,
        }
        cluster.size = len(indices)
        cluster.confidence = confidence
        await db.flush()
        representative_index = indices[similarities.index(max(similarities))]
        norm = await db.scalar(
            select(CandidateNorm).where(CandidateNorm.cluster_id == cluster.id)
        )
        if norm is None:
            norm = CandidateNorm(
                repository_id=repository_id,
                cluster_id=cluster.id,
                statement=comments[representative_index].normalized_body or "",
                confidence=confidence,
                support_count=len(indices),
            )
            db.add(norm)
        else:
            norm.support_count = len(indices)
        await db.flush()
        for index, similarity in zip(indices, similarities):
            comment = comments[index]
            db.add(
                CommentClusterMember(
                    cluster_id=cluster.id,
                    review_comment_id=comment.id,
                    similarity_to_centroid=similarity,
                )
            )
            evidence = await db.scalar(
                select(CandidateNormEvidence).where(
                    CandidateNormEvidence.candidate_norm_id == norm.id,
                    CandidateNormEvidence.review_comment_id == comment.id,
                )
            )
            if evidence is None:
                db.add(
                    CandidateNormEvidence(
                        candidate_norm_id=norm.id,
                        review_comment_id=comment.id,
                        pull_request_id=comment.pull_request_id,
                    )
                )
        norm_count += 1

    await db.commit()
    return MiningResult(
        len(comments),
        len(comments),
        len(grouped),
        norm_count,
        eligible=len(comments),
        excluded=excluded,
        noise=noise,
        embeddings_reused=reused,
        embeddings_created=len(missing_comments),
    )
