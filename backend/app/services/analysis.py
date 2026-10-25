"""Explainable, provisional Phase 4 pull-request norm analysis."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    CandidateNorm,
    CandidateNormEvidence,
    CommentEmbedding,
    NormFinding,
    NormFindingFeedback,
    PullRequest,
    PullRequestAnalysis,
    PullRequestFile,
    ReviewComment,
)
from app.services.phase2 import cosine_similarity

FORMULA_VERSION = "v2"
RISK_FORMULA = (
    "risk_score = 0.45 * relevance_score + 0.35 * evidence_strength "
    "+ 0.20 * norm_confidence; risk levels: high >= 0.80, medium >= 0.60, "
    "low otherwise. A violation is suspected when fewer than half of the "
    "norm's key terms appear in the added lines. Findings are always provisional."
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "with",
    "this",
    "that",
    "must",
    "should",
    "please",
    "here",
    "there",
    "can",
    "could",
    "would",
    "need",
    "needs",
    "add",
    "make",
    "sure",
    "always",
    "never",
    "not",
    "don",
    "instead",
    "when",
    "it",
    "we",
    "you",
    "our",
    "your",
}


class Embedder(Protocol):
    model_name: str

    def encode(
        self, texts: Sequence[str], batch_size: int
    ) -> Sequence[Sequence[float]]: ...


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9_]{2,}", text.casefold())
        if token not in STOP_WORDS
    }


def _added_text(patch: str) -> str:
    """Lines a patch adds; removed and context lines say nothing about the change."""
    return "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


CODE_WINDOW_LINES = 3


def _added_windows(patch: str, size: int = CODE_WINDOW_LINES) -> list[tuple[int, int, str]]:
    """Sliding windows of consecutive added lines as (first line, last line, text)."""
    added: list[tuple[int, str]] = []
    new_line = 0
    for line in patch.splitlines():
        header = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
        if header:
            new_line = int(header.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            added.append((new_line, line[1:]))
            new_line += 1
        elif not line.startswith("-"):
            new_line += 1
    windows = []
    for start in range(max(1, len(added) - size + 1)):
        chunk = added[start : start + size]
        if chunk and any(text.strip() for _, text in chunk):
            windows.append((chunk[0][0], chunk[-1][0], "\n".join(text for _, text in chunk)))
    return windows


def _hunk_tail(diff_hunk: str, size: int = CODE_WINDOW_LINES) -> str:
    """The lines a review comment was attached to: the end of its diff hunk."""
    lines = [line[1:] for line in diff_hunk.splitlines() if not line.startswith("@@")]
    return "\n".join(lines[-size:])


def _line_range(patch: str | None) -> tuple[int | None, int | None]:
    if not patch:
        return None, None
    hunks = re.findall(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))?", patch)
    if not hunks:
        return None, None
    start = int(hunks[0][0])
    last_start, last_count = hunks[-1]
    return start, max(start, int(last_start) + int(last_count or 1) - 1)


def _risk_level(score: float, suspected: bool) -> str:
    if not suspected:
        return "none"
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


RESULT_FIELDS = (
    "pull_request_file_id",
    "line_end",
    "relevance_score",
    "evidence_strength",
    "risk_score",
    "risk_level",
    "provisional",
    "violation_suspected",
    "explanation",
    "evidence_comment_ids",
    "evidence_pull_request_ids",
)


async def _reconcile_findings(
    db: AsyncSession,
    existing: Sequence[NormFinding],
    labeled_ids: set[int],
    new: Sequence[NormFinding],
) -> list[NormFinding]:
    """Replace a previous analysis's findings without losing human feedback.

    A finding for the same norm and location is updated in place, keeping its
    feedback and status. Previous findings that no longer apply are deleted,
    unless they carry feedback: evaluation still relies on those, so they are
    kept but marked as no longer current.
    """
    by_key = {
        (finding.candidate_norm_id, finding.file_path, finding.line_start): finding
        for finding in existing
    }
    current: list[NormFinding] = []
    for finding in new:
        match = by_key.pop(
            (finding.candidate_norm_id, finding.file_path, finding.line_start), None
        )
        if match is None:
            db.add(finding)
            current.append(finding)
            continue
        for field in RESULT_FIELDS:
            setattr(match, field, getattr(finding, field))
        match.current = True
        current.append(match)
    for stale in by_key.values():
        if stale.id in labeled_ids:
            stale.current = False
        else:
            await db.delete(stale)
    return current


@dataclass(frozen=True)
class AnalysisResult:
    analysis_id: int
    findings: int
    context_available: bool
    highest_risk: str


async def analyze_pull_request(
    db: AsyncSession,
    pull_request_id: int,
    *,
    embedder: Embedder,
    batch_size: int = 32,
    relevance_threshold: float = 0.55,
) -> AnalysisResult:
    pull_request = await db.get(PullRequest, pull_request_id)
    if pull_request is None:
        raise ValueError(f"pull request {pull_request_id} not found")
    files = list(
        (
            await db.scalars(
                select(PullRequestFile)
                .where(PullRequestFile.pull_request_id == pull_request_id)
                .order_by(PullRequestFile.id)
            )
        ).all()
    )
    contexts = [f"{item.filename}\n{item.patch or ''}" for item in files if item.patch]
    analysis = await db.scalar(
        select(PullRequestAnalysis).where(
            PullRequestAnalysis.pull_request_id == pull_request_id
        )
    )
    existing_findings: list[NormFinding] = []
    labeled_ids: set[int] = set()
    if analysis is None:
        analysis = PullRequestAnalysis(pull_request_id=pull_request_id)
        db.add(analysis)
        await db.flush()
    else:
        existing_findings = list(
            (
                await db.scalars(
                    select(NormFinding).where(NormFinding.analysis_id == analysis.id)
                )
            ).all()
        )
        if existing_findings:
            labeled_ids = set(
                (
                    await db.scalars(
                        select(NormFindingFeedback.finding_id).where(
                            NormFindingFeedback.finding_id.in_(
                                [finding.id for finding in existing_findings]
                            )
                        )
                    )
                ).all()
            )
    analysis.status = "completed"
    analysis.formula_version = FORMULA_VERSION
    analysis.context_available = bool(contexts)
    if not contexts:
        await _reconcile_findings(db, existing_findings, labeled_ids, [])
        analysis.findings_count = 0
        analysis.highest_risk = "none"
        await db.commit()
        return AnalysisResult(analysis.id, 0, False, "none")

    vectors = list(embedder.encode(contexts, batch_size=batch_size))
    if len(vectors) != len(contexts):
        raise ValueError("embedding provider returned an unexpected vector count")
    approved_norms = list(
        (
            await db.scalars(
                select(CandidateNorm).where(
                    CandidateNorm.repository_id == pull_request.repository_id,
                    CandidateNorm.approved.is_(True),
                    CandidateNorm.status == "active",
                )
            )
        ).all()
    )
    evidence_rows = list(
        (
            await db.scalars(
                select(CandidateNormEvidence).where(
                    CandidateNormEvidence.candidate_norm_id.in_(
                        [norm.id for norm in approved_norms]
                    ),
                    CandidateNormEvidence.active.is_(True),
                )
            )
        ).all()
    )
    by_norm: dict[int, list[CandidateNormEvidence]] = {}
    for evidence in evidence_rows:
        by_norm.setdefault(evidence.candidate_norm_id, []).append(evidence)
    embeddings_by_comment = {
        embedding.review_comment_id: embedding
        for embedding in (
            await db.scalars(
                select(CommentEmbedding).where(
                    CommentEmbedding.review_comment_id.in_(
                        [item.review_comment_id for item in evidence_rows]
                    ),
                    CommentEmbedding.model_name == embedder.model_name,
                )
            )
        ).all()
    }
    # Code that drew each review comment, so new code can be compared with
    # code reviewers flagged, not only with the wording of their comments.
    hunk_comments = [
        comment
        for comment in (
            await db.scalars(
                select(ReviewComment).where(
                    ReviewComment.id.in_([item.review_comment_id for item in evidence_rows]),
                    ReviewComment.diff_hunk.is_not(None),
                )
            )
        ).all()
        if comment.diff_hunk
    ]
    hunk_vectors: dict[int, Sequence[float]] = {}
    if hunk_comments:
        encoded = list(
            embedder.encode(
                [_hunk_tail(item.diff_hunk) for item in hunk_comments],
                batch_size=batch_size,
            )
        )
        hunk_vectors = {item.id: vector for item, vector in zip(hunk_comments, encoded)}
    patched_files = [item for item in files if item.patch]
    windows_by_file = {item.id: _added_windows(item.patch) for item in patched_files}
    flat_windows = [window for item in patched_files for window in windows_by_file[item.id]]
    window_vectors = (
        list(embedder.encode([text for _, _, text in flat_windows], batch_size=batch_size))
        if hunk_vectors and flat_windows
        else []
    )
    vectors_by_file: dict[int, list[tuple[int, int, Sequence[float]]]] = {}
    position = 0
    for item in patched_files:
        count = len(windows_by_file[item.id]) if window_vectors else 0
        vectors_by_file[item.id] = [
            (start, end, window_vectors[position + index])
            for index, (start, end, _) in enumerate(windows_by_file[item.id][:count])
        ]
        position += count
    findings: list[NormFinding] = []
    for file, context, vector in zip(
        [item for item in files if item.patch], contexts, vectors
    ):
        for norm in approved_norms:
            evidence = by_norm.get(norm.id, [])
            comment_ids = [item.review_comment_id for item in evidence]
            embeddings = [
                embeddings_by_comment[comment_id]
                for comment_id in comment_ids
                if comment_id in embeddings_by_comment
            ]
            if not embeddings:
                continue
            text_relevance = max(
                cosine_similarity(vector, embedding.vector) for embedding in embeddings
            )
            # Best match between a few added lines and the code a reviewer flagged.
            code_relevance, best_window = 0.0, None
            for start, end, window_vector in vectors_by_file.get(file.id, []):
                for comment_id in comment_ids:
                    if comment_id in hunk_vectors:
                        score = cosine_similarity(window_vector, hunk_vectors[comment_id])
                        if score > code_relevance:
                            code_relevance, best_window = score, (start, end)
            relevance = max(text_relevance, code_relevance)
            if relevance < relevance_threshold:
                continue
            evidence_strength = min(1.0, len(evidence) / 5)
            evidence_pr_ids = sorted(
                {item.pull_request_id for item in evidence if item.pull_request_id}
            )
            norm_tokens = _tokens(norm.statement)
            context_tokens = _tokens(f"{file.filename}\n{_added_text(file.patch)}")
            missing_tokens = norm_tokens - context_tokens
            # One generic missing word should not flag a change; most key terms must be.
            suspected = bool(norm_tokens) and len(missing_tokens) * 2 > len(norm_tokens)
            score = min(
                1.0,
                0.45 * relevance + 0.35 * evidence_strength + 0.20 * norm.confidence,
            )
            explanation = (
                f"Relevant approved norm matched with relevance {relevance:.2f} "
                f"(review-comment text {text_relevance:.2f}, previously reviewed code "
                f"{code_relevance:.2f}); {len(evidence)} historical comment(s) support it. "
            )
            if suspected:
                explanation += (
                    "Potential violation: most key norm terms are absent from the "
                    f"added lines ({', '.join(sorted(missing_tokens))})."
                )
            else:
                explanation += "No deterministic contradiction signal was found."
            line_start, line_end = (
                best_window
                if best_window and code_relevance >= text_relevance
                else _line_range(file.patch)
            )
            findings.append(
                NormFinding(
                    analysis_id=analysis.id,
                    candidate_norm_id=norm.id,
                    pull_request_file_id=file.id,
                    file_path=file.filename,
                    line_start=line_start,
                    line_end=line_end,
                    relevance_score=relevance,
                    evidence_strength=evidence_strength,
                    risk_score=score,
                    risk_level=_risk_level(score, suspected),
                    provisional=True,
                    violation_suspected=suspected,
                    explanation=explanation,
                    evidence_comment_ids=comment_ids,
                    evidence_pull_request_ids=evidence_pr_ids,
                )
            )
    findings = await _reconcile_findings(db, existing_findings, labeled_ids, findings)
    analysis.findings_count = len(findings)
    analysis.highest_risk = max(
        (finding.risk_level for finding in findings),
        key={"none": 0, "low": 1, "medium": 2, "high": 3}.get,
        default="none",
    )
    await db.commit()
    return AnalysisResult(analysis.id, len(findings), True, analysis.highest_risk)


async def record_feedback(
    db: AsyncSession,
    finding_id: int,
    *,
    decision: str,
    note: str | None,
    changed_by: str | None,
) -> NormFindingFeedback:
    if decision not in {"confirmed", "dismissed", "accepted", "needs_review"}:
        raise ValueError("invalid feedback decision")
    finding = await db.get(NormFinding, finding_id)
    if finding is None:
        raise ValueError(f"finding {finding_id} not found")
    feedback = NormFindingFeedback(
        finding_id=finding_id,
        decision=decision,
        note=note,
        changed_by=changed_by,
    )
    db.add(feedback)
    finding.status = decision
    await db.commit()
    await db.refresh(feedback)
    return feedback
