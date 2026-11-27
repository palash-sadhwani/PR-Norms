"""Deterministic semantic embedding, clustering, and candidate norm mining."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
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
    User,
)
from app.services.norms import ensure_initial_version, record_version
from app.services.preprocess import preprocess_comment

# A new cluster continues a previous one when at least half of the smaller
# cluster's comments are shared, or when their centroids nearly coincide.
MATCH_MIN_OVERLAP = 0.5
MATCH_MIN_CENTROID_SIMILARITY = 0.9
# Stored under its own model name so fallback vectors never mix with real ones.
DEMO_FALLBACK_MODEL = "demo-fallback/hashing-bag-of-words-v1"
# Below this many comments HDBSCAN's default min_samples marks nearly
# everything as noise, so a configured-unset min_samples drops to 1.
SMALL_DATASET_COMMENTS = 50
KEYWORD_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "because",
        "before",
        "being",
        "change",
        "changes",
        "code",
        "consider",
        "could",
        "doing",
        "file",
        "here",
        "into",
        "just",
        "keep",
        "keeping",
        "make",
        "maybe",
        "please",
        "rather",
        "should",
        "that",
        "this",
        "use",
        "using",
        "would",
    }
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
    total_imported: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    recovered_comments: int = 0
    hdbscan_clusters: int = 0
    rejected_clusters: int = 0
    rejected_insufficient_evidence: int = 0
    rejected_insufficient_pr_diversity: int = 0
    candidate_norms_generated: int = 0
    zero_norm_explanation: str | None = None


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


def _meaningful_keywords(text: str) -> frozenset[str]:
    words = set()
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        if len(token) < 3 or token in KEYWORD_STOPWORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and not token.endswith(("ss", "us")):
            token = token[:-1]
        words.add(token)
    return frozenset(words)


def _keyword_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _recover_noise_groups(
    noise_indices: Sequence[int],
    comments: Sequence[ReviewComment],
    normalized_text: dict[int, str],
    vectors: Sequence[Sequence[float]],
    *,
    min_cosine: float,
    min_keyword_overlap: float,
) -> list[list[int]]:
    """Recover only complete-link groups; a weak bridge cannot chain topics."""
    keywords = {
        index: _meaningful_keywords(normalized_text[comments[index].id])
        for index in noise_indices
    }
    keyword_indices: dict[str, list[int]] = {}
    for index, terms in keywords.items():
        for term in terms:
            keyword_indices.setdefault(term, []).append(index)
    possible_pairs = {
        (left, right)
        for indices in keyword_indices.values()
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
    }

    candidates: list[tuple[float, int, int]] = []
    compatible_pairs: set[tuple[int, int]] = set()
    for left, right in sorted(possible_pairs):
        similarity = cosine_similarity(vectors[left], vectors[right])
        if (
            similarity >= min_cosine
            and _keyword_overlap(keywords[left], keywords[right])
            >= min_keyword_overlap
        ):
            compatible_pairs.add((left, right))
            candidates.append((similarity, left, right))

    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    groups = {index: {index} for index in noise_indices if keywords[index]}
    group_for = {index: index for index in groups}
    for _, left, right in candidates:
        left_group = group_for[left]
        right_group = group_for[right]
        if left_group == right_group:
            continue
        left_members = groups[left_group]
        right_members = groups[right_group]
        if not all(
            (min(first, second), max(first, second)) in compatible_pairs
            for first in left_members
            for second in right_members
        ):
            continue
        merged = left_members | right_members
        root = min(left_group, right_group)
        discard = max(left_group, right_group)
        groups[root] = merged
        del groups[discard]
        for index in merged:
            group_for[index] = root

    return sorted(
        (sorted(members) for members in groups.values() if len(members) >= 2),
        key=lambda members: members[0],
    )


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
                )
                .order_by(ReviewComment.id)
            )
        ).all()
    )
    total_imported = len(comments)
    issue_comment_ids = set(
        (
            await db.scalars(
                select(IssueComment.id)
                .join(PullRequest, PullRequest.id == IssueComment.pull_request_id)
                .where(PullRequest.repository_id == repository_id)
            )
        ).all()
    )
    author_ids = {comment.author_id for comment in comments if comment.author_id}
    author_logins = (
        dict(
            (
                await db.execute(
                    select(User.id, User.login).where(User.id.in_(author_ids))
                )
            ).all()
        )
        if author_ids
        else {}
    )
    exclusion_reasons: Counter[str] = Counter()
    eligible_comments: list[ReviewComment] = []
    normalized_text: dict[int, str] = {}
    for comment in comments:
        reason = None
        if comment.id in issue_comment_ids:
            reason = "general_pr_conversation"
        elif comment.is_filtered:
            reason = comment.filter_reason or "filtered"
        else:
            processed = preprocess_comment(
                comment.body, author_logins.get(comment.author_id)
            )
            if processed.is_low_information:
                reason = processed.reason or "low_information"
            elif not comment.normalized_body:
                reason = "missing_normalized_body"
            elif not comment.path or not any(
                (
                    comment.line is not None,
                    comment.original_line is not None,
                    comment.start_line is not None,
                    comment.position is not None,
                    comment.diff_hunk,
                )
            ):
                reason = "not_inline_review_comment"
        if reason:
            exclusion_reasons[reason] += 1
            continue
        eligible_comments.append(comment)
        normalized_text[comment.id] = preprocess_comment(
            comment.body, author_logins.get(comment.author_id)
        ).normalized_text

    comments = eligible_comments
    excluded = total_imported - len(comments)
    model_name = embedder.model_name if embedder else model_name
    existing_embeddings: dict[int, CommentEmbedding] = {}
    vectors_by_comment: dict[int, Sequence[float]] = {}
    missing_comments: list[ReviewComment] = []
    reused = 0
    if comments:
        embedder = embedder or create_embedder(model_name)
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
            comment.id: _hash_text(normalized_text[comment.id]) for comment in comments
        }
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
                [normalized_text[comment.id] for comment in missing_comments],
                batch_size=batch_size,
            )
            if len(generated) != len(missing_comments):
                raise ValueError("embedding provider returned an unexpected vector count")
            for comment, vector in zip(missing_comments, generated):
                vectors_by_comment[comment.id] = vector
        vectors = [vectors_by_comment[comment.id] for comment in comments]
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
            embedding.source_text_hash = _hash_text(normalized_text[comment.id])
            embedding.dimensions = len(vector)
            embedding.vector = [float(value) for value in vector]
    else:
        vectors = []

    min_samples = settings.phase2_min_samples
    if min_samples is None and len(vectors) < SMALL_DATASET_COMMENTS:
        min_samples = 1
    if not vectors or (clusterer is None and len(vectors) < settings.phase2_min_cluster_size):
        labels = [-1] * len(vectors)
    else:
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

    hdbscan_grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label >= 0:
            hdbscan_grouped.setdefault(int(label), []).append(index)
    hdbscan_clusters = len(hdbscan_grouped)
    rejected_insufficient_evidence = 0
    rejected_insufficient_pr_diversity = 0
    accepted_groups: dict[int, list[int]] = {}
    for label, indices in hdbscan_grouped.items():
        if len(indices) < settings.phase2_min_support_comments:
            rejected_insufficient_evidence += 1
        elif len({comments[index].pull_request_id for index in indices}) < (
            settings.phase2_min_support_pull_requests
        ):
            rejected_insufficient_pr_diversity += 1
        else:
            accepted_groups[label] = indices

    noise_indices = [index for index, label in enumerate(labels) if label == -1]
    recovery_candidates = _recover_noise_groups(
        noise_indices,
        comments,
        normalized_text,
        vectors,
        min_cosine=settings.phase2_recovery_min_cosine,
        min_keyword_overlap=settings.phase2_recovery_min_keyword_overlap,
    )
    recovered_groups = [
        indices for indices in recovery_candidates if len(indices) >= 3
    ]
    rejected_insufficient_evidence += sum(
        len(indices) < settings.phase2_min_support_comments
        for indices in recovery_candidates
    )
    next_label = max(hdbscan_grouped, default=-1) + 1
    recovered_comments = sum(len(indices) for indices in recovered_groups)
    for indices in recovered_groups:
        if len(indices) < settings.phase2_min_support_comments:
            continue
        if len({comments[index].pull_request_id for index in indices}) < (
            settings.phase2_min_support_pull_requests
        ):
            rejected_insufficient_pr_diversity += 1
        else:
            accepted_groups[next_label] = indices
        next_label += 1
    rejected_clusters = (
        rejected_insufficient_evidence + rejected_insufficient_pr_diversity
    )
    grouped = accepted_groups
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
    noise = len(noise_indices) - recovered_comments
    for label, indices in sorted(grouped.items()):
        similarities = [
            cosine_similarity(vectors[index], centers[label]) for index in indices
        ]
        distinct_pull_requests = len(
            {comments[index].pull_request_id for index in indices}
        )
        confidence = (
            0.6 * (sum(similarities) / len(similarities))
            + 0.25 * min(len(indices) / 6, 1.0)
            + 0.15 * min(distinct_pull_requests / 3, 1.0)
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
            "min_samples": min_samples,
            "random_seed": settings.phase2_random_seed,
            "min_support_comments": settings.phase2_min_support_comments,
            "min_support_pull_requests": settings.phase2_min_support_pull_requests,
            "recovery_min_cosine": settings.phase2_recovery_min_cosine,
            "recovery_min_keyword_overlap": settings.phase2_recovery_min_keyword_overlap,
            "recovered_from_noise": label not in hdbscan_grouped,
        }
        cluster.size = len(indices)
        cluster.confidence = confidence
        await db.flush()
        representative_index = indices[similarities.index(max(similarities))]
        representative = normalized_text[comments[representative_index].id]
        norm = await db.scalar(
            select(CandidateNorm).where(CandidateNorm.cluster_id == cluster.id)
        )
        created = norm is None
        if norm is None:
            norm = CandidateNorm(
                repository_id=repository_id,
                cluster_id=cluster.id,
                statement=representative,
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
                norm.statement = representative or norm.statement
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
            norm.support_count = 0
        await record_version(
            db,
            norm,
            reason="Mining no longer finds this norm's cluster; review it before relying on it.",
            changed_by=None,
        )
        norms_retired += 1

    await db.commit()
    if norm_count:
        zero_norm_explanation = None
    elif not comments:
        zero_norm_explanation = "No eligible inline review comments were found."
    elif len(comments) < settings.phase2_min_support_comments:
        zero_norm_explanation = (
            "Not enough comments exist to satisfy the minimum support requirement."
        )
    elif rejected_insufficient_pr_diversity:
        zero_norm_explanation = (
            "Clusters did not have evidence across enough distinct pull requests."
        )
    elif rejected_insufficient_evidence:
        zero_norm_explanation = "Clusters did not meet the minimum comment support."
    else:
        zero_norm_explanation = (
            "Comments were too dissimilar to form meaningful clusters."
        )
    return MiningResult(
        len(comments),
        sum(len(indices) for indices in grouped.values()),
        len(grouped),
        norm_count,
        eligible=len(comments),
        excluded=excluded,
        noise=noise,
        embeddings_reused=reused,
        embeddings_created=len(missing_comments),
        norms_created=norms_created,
        norms_retired=norms_retired,
        total_imported=total_imported,
        exclusion_reasons=dict(exclusion_reasons),
        recovered_comments=recovered_comments,
        hdbscan_clusters=hdbscan_clusters,
        rejected_clusters=rejected_clusters,
        rejected_insufficient_evidence=rejected_insufficient_evidence,
        rejected_insufficient_pr_diversity=rejected_insufficient_pr_diversity,
        candidate_norms_generated=norm_count,
        zero_norm_explanation=zero_norm_explanation,
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
