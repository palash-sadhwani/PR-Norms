"""Deterministic semantic embedding, clustering, and candidate norm mining."""

from __future__ import annotations

import hashlib
import math
import re
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
    IssueComment,
    NormVersion,
    PullRequest,
    Repository,
    ReviewComment,
)
from app.services.norms import ensure_initial_version, record_version

# A new cluster continues a previous one when at least half of the smaller
# cluster's comments are shared, or when their centroids nearly coincide.
MATCH_MIN_OVERLAP = 0.5
MATCH_MIN_CENTROID_SIMILARITY = 0.9
# Stored under its own model name so fallback vectors never mix with real ones.
DEMO_FALLBACK_MODEL = "demo-fallback/hashing-bag-of-words-v1"
# Below this many comments HDBSCAN's default min_samples marks nearly
# everything as noise, so a configured-unset min_samples drops to 1.
SMALL_DATASET_COMMENTS = 50


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
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            # Download/proxy/cache failures surface as many exception types.
            raise RuntimeError(
                f"Embedding model {model_name!r} could not be loaded: {exc}"
            ) from exc

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


class HashingEmbedder:
    """Lexical bag-of-words vectors for offline demos. Not a semantic model."""

    dimensions = 512
    model_name = DEMO_FALLBACK_MODEL

    def encode(
        self, texts: Sequence[str], batch_size: int
    ) -> Sequence[Sequence[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            # Split identifiers such as api_key into words.
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if len(token) > 2:
                    digest = hashlib.sha256(token.encode()).digest()
                    vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def is_fallback_model(model_name: str | None) -> bool:
    return model_name == DEMO_FALLBACK_MODEL


def create_embedder(model_name: str | None = None) -> Embedder:
    settings = get_settings()
    if settings.phase2_embedding_backend == "demo-hashing":
        return HashingEmbedder()
    if settings.phase2_embedding_backend != "sentence-transformers":
        raise RuntimeError(
            f"Unknown PHASE2_EMBEDDING_BACKEND {settings.phase2_embedding_backend!r}"
        )
    return SentenceTransformerEmbedder(model_name or settings.phase2_embedding_model)


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
    norms_created: int = 0
    norms_retired: int = 0


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
                    # Norms come from code-review comments, not general PR conversation.
                    ReviewComment.id.not_in(select(IssueComment.id)),
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

    embedder = embedder or create_embedder(model_name)
    # Store vectors under the name analysis will look them up by.
    model_name = embedder.model_name
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

    min_samples = settings.phase2_min_samples
    if min_samples is None and len(vectors) < SMALL_DATASET_COMMENTS:
        min_samples = 1
    clusterer = clusterer or HdbscanClusterer(
        settings.phase2_min_cluster_size,
        min_samples,
        settings.phase2_random_seed,
    )
    labels = list(clusterer.fit_predict(vectors))
    if len(labels) != len(comments):
        raise ValueError("clusterer returned an unexpected label count")

    previous_clusters = list(
        (
            await db.scalars(
                select(CommentCluster).where(
                    CommentCluster.repository_id == repository_id,
                    CommentCluster.model_name == model_name,
                    CommentCluster.algorithm == "hdbscan",
                    CommentCluster.active.is_(True),
                )
            )
        ).all()
    )
    previous_members: dict[int, set[int]] = {
        cluster.id: set() for cluster in previous_clusters
    }
    if previous_clusters:
        for cluster_id, comment_id in (
            await db.execute(
                select(
                    CommentClusterMember.cluster_id,
                    CommentClusterMember.review_comment_id,
                ).where(CommentClusterMember.cluster_id.in_(previous_members))
            )
        ).all():
            previous_members[cluster_id].add(comment_id)
        # Clusters recorded without member rows (older data, seeded demos) are
        # identified by their norm's active evidence instead.
        unrecorded = [cid for cid, members in previous_members.items() if not members]
        if unrecorded:
            for cluster_id, comment_id in (
                await db.execute(
                    select(CandidateNorm.cluster_id, CandidateNormEvidence.review_comment_id)
                    .join(
                        CandidateNormEvidence,
                        CandidateNormEvidence.candidate_norm_id == CandidateNorm.id,
                    )
                    .where(
                        CandidateNorm.cluster_id.in_(unrecorded),
                        CandidateNormEvidence.active.is_(True),
                    )
                )
            ).all():
                previous_members[cluster_id].add(comment_id)
        await db.execute(
            delete(CommentClusterMember).where(
                CommentClusterMember.cluster_id.in_(previous_members)
            )
        )
    # Labels are per-run numbers; park previous clusters on unique negative
    # labels so this run's labels can be assigned without collisions.
    for cluster in previous_clusters:
        cluster.label = -cluster.id
    await db.flush()

    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label >= 0:
            grouped.setdefault(int(label), []).append(index)
    centers = {
        label: _centroid([vectors[index] for index in indices])
        for label, indices in grouped.items()
    }
    matches = _match_clusters(
        {
            label: {comments[index].id for index in indices}
            for label, indices in grouped.items()
        },
        centers,
        previous_members,
        vectors_by_comment,
    )
    clusters_by_id = {cluster.id: cluster for cluster in previous_clusters}

    norm_count = 0
    norms_created = 0
    noise = labels.count(-1)
    for label, indices in sorted(grouped.items()):
        similarities = [
            cosine_similarity(vectors[index], centers[label]) for index in indices
        ]
        confidence = min(
            1.0,
            (len(indices) / max(len(comments), 1)) * 0.5
            + (sum(similarities) / len(similarities)) * 0.5,
        )
        cluster = clusters_by_id.get(matches.get(label, -1))
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
        cluster.label = label
        cluster.active = True
        cluster.algorithm_config = {
            "min_cluster_size": settings.phase2_min_cluster_size,
            "min_samples": settings.phase2_min_samples,
            "random_seed": settings.phase2_random_seed,
        }
        cluster.size = len(indices)
        cluster.confidence = confidence
        await db.flush()
        representative = comments[indices[similarities.index(max(similarities))]]
        norm = await db.scalar(
            select(CandidateNorm).where(CandidateNorm.cluster_id == cluster.id)
        )
        created = norm is None
        if norm is None:
            norm = CandidateNorm(
                repository_id=repository_id,
                cluster_id=cluster.id,
                statement=representative.normalized_body or "",
                confidence=confidence,
                support_count=len(indices),
            )
            db.add(norm)
            norms_created += 1
        norm.support_count = len(indices)
        await db.flush()
        for index, similarity in zip(indices, similarities):
            db.add(
                CommentClusterMember(
                    cluster_id=cluster.id,
                    review_comment_id=comments[index].id,
                    similarity_to_centroid=similarity,
                )
            )
        added, removed = await _sync_evidence(
            db, norm, [comments[index] for index in indices]
        )
        if not created:
            if not await _has_history(db, norm.id):
                # Untouched machine candidate: keep it describing its cluster.
                norm.statement = representative.normalized_body or norm.statement
                norm.confidence = confidence
            elif added or removed:
                await record_version(
                    db,
                    norm,
                    reason=(
                        f"Mining run updated evidence: {added} comment(s) joined, "
                        f"{removed} no longer clustered with this norm."
                    ),
                    changed_by=None,
                )
        norm_count += 1

    norms_retired = 0
    matched_ids = set(matches.values())
    for cluster in previous_clusters:
        if cluster.id in matched_ids:
            continue
        cluster.active = False
        cluster.size = 0
        norm = await db.scalar(
            select(CandidateNorm).where(CandidateNorm.cluster_id == cluster.id)
        )
        if norm is None:
            continue
        await ensure_initial_version(db, norm)
        await _sync_evidence(db, norm, [])
        if norm.status in {"emerging", "active"}:
            norm.status = "weakening"
        await record_version(
            db,
            norm,
            reason="Mining no longer finds this norm's cluster; review it before relying on it.",
            changed_by=None,
        )
        norms_retired += 1

    await db.commit()
    return MiningResult(
        len(comments),
        len(comments) - noise,
        len(grouped),
        norm_count,
        eligible=len(comments),
        excluded=excluded,
        noise=noise,
        embeddings_reused=reused,
        embeddings_created=len(missing_comments),
        norms_created=norms_created,
        norms_retired=norms_retired,
    )


def _match_clusters(
    groups: dict[int, set[int]],
    centers: dict[int, Sequence[float]],
    previous_members: dict[int, set[int]],
    vectors_by_comment: dict[int, Sequence[float]],
) -> dict[int, int]:
    """Pair this run's cluster labels with previous cluster ids, one to one.

    HDBSCAN labels are arbitrary per run, so identity comes from shared member
    comments, falling back to centroid similarity when a cluster grew or
    shifted enough that overlap alone is inconclusive.
    """
    previous_centers = {}
    for cluster_id, members in previous_members.items():
        present = [vectors_by_comment[i] for i in members if i in vectors_by_comment]
        if present:
            previous_centers[cluster_id] = _centroid(present)
    candidates = []
    for label, members in groups.items():
        for cluster_id, previous in previous_members.items():
            overlap = (
                len(members & previous) / min(len(members), len(previous))
                if previous
                else 0.0
            )
            similarity = (
                cosine_similarity(centers[label], previous_centers[cluster_id])
                if cluster_id in previous_centers
                else 0.0
            )
            if (
                overlap >= MATCH_MIN_OVERLAP
                or similarity >= MATCH_MIN_CENTROID_SIMILARITY
            ):
                candidates.append((overlap, similarity, label, cluster_id))
    matches: dict[int, int] = {}
    for _, _, label, cluster_id in sorted(candidates, reverse=True):
        if label not in matches and cluster_id not in matches.values():
            matches[label] = cluster_id
    return matches


async def _sync_evidence(
    db: AsyncSession, norm: CandidateNorm, members: Sequence[ReviewComment]
) -> tuple[int, int]:
    """Make the norm's active evidence equal its cluster members.

    Evidence rows are never deleted; comments that leave the cluster are
    marked inactive so the norm's history stays traceable.
    """
    existing = {
        evidence.review_comment_id: evidence
        for evidence in (
            await db.scalars(
                select(CandidateNormEvidence).where(
                    CandidateNormEvidence.candidate_norm_id == norm.id
                )
            )
        ).all()
    }
    member_ids = {comment.id for comment in members}
    added = removed = 0
    for comment in members:
        evidence = existing.get(comment.id)
        if evidence is None:
            db.add(
                CandidateNormEvidence(
                    candidate_norm_id=norm.id,
                    review_comment_id=comment.id,
                    pull_request_id=comment.pull_request_id,
                )
            )
            added += 1
        elif not evidence.active:
            evidence.active = True
            added += 1
    for comment_id, evidence in existing.items():
        if comment_id not in member_ids and evidence.active:
            evidence.active = False
            removed += 1
    await db.flush()
    return added, removed


async def _has_history(db: AsyncSession, norm_id: int) -> bool:
    return bool(
        await db.scalar(
            select(func.count(NormVersion.id)).where(
                NormVersion.candidate_norm_id == norm_id
            )
        )
    )
