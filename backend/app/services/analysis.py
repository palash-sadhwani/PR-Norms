"""Explainable, provisional Phase 4 pull-request norm analysis."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete, select
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
)
from app.services.phase2 import cosine_similarity

FORMULA_VERSION = "v1"
RISK_FORMULA = (
    "risk_score = 0.45 * relevance_score + 0.35 * evidence_strength "
    "+ 0.20 * norm_confidence; risk levels: high >= 0.80, medium >= 0.60, "
    "low otherwise. Findings are always provisional."
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


def _line_range(patch: str | None) -> tuple[int | None, int | None]:
    if not patch:
        return None, None
    match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))?", patch)
    if not match:
        return None, None
    start = int(match.group(1))
    count = int(match.group(2) or 1)
    return start, start + count - 1


def _risk_level(score: float, suspected: bool) -> str:
    if not suspected:
        return "none"
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


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
    if analysis is None:
        analysis = PullRequestAnalysis(pull_request_id=pull_request_id)
        db.add(analysis)
        await db.flush()
    else:
        await db.execute(
            delete(NormFinding).where(NormFinding.analysis_id == analysis.id)
        )
    analysis.status = "completed"
    analysis.formula_version = FORMULA_VERSION
    analysis.context_available = bool(contexts)
    if not contexts:
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
    findings: list[NormFinding] = []
    for file, context, vector in zip(
        [item for item in files if item.patch], contexts, vectors
    ):
        evidence_rows = list(
            (
                await db.scalars(
                    select(CandidateNormEvidence).where(
                        CandidateNormEvidence.candidate_norm_id.in_(
                            [norm.id for norm in approved_norms]
                        )
                    )
                )
            ).all()
        )
        by_norm: dict[int, list[CandidateNormEvidence]] = {}
        for evidence in evidence_rows:
            by_norm.setdefault(evidence.candidate_norm_id, []).append(evidence)
        for norm in approved_norms:
            evidence = by_norm.get(norm.id, [])
            comment_ids = [item.review_comment_id for item in evidence]
            embeddings = list(
                (
                    await db.scalars(
                        select(CommentEmbedding).where(
                            CommentEmbedding.review_comment_id.in_(comment_ids),
                            CommentEmbedding.model_name == embedder.model_name,
                        )
                    )
                ).all()
            )
            if not embeddings:
                continue
            relevance = max(
                cosine_similarity(vector, embedding.vector) for embedding in embeddings
            )
            if relevance < relevance_threshold:
                continue
            evidence_strength = min(1.0, len(evidence) / 5)
            evidence_pr_ids = sorted(
                {item.pull_request_id for item in evidence if item.pull_request_id}
            )
            norm_tokens = _tokens(norm.statement)
            context_tokens = _tokens(context)
            missing_tokens = norm_tokens - context_tokens
            suspected = bool(missing_tokens) and relevance >= relevance_threshold
            score = min(
                1.0,
                0.45 * relevance + 0.35 * evidence_strength + 0.20 * norm.confidence,
            )
            explanation = (
                f"Relevant approved norm matched with semantic relevance "
                f"{relevance:.2f}; {len(evidence)} historical comment(s) support it. "
            )
            if suspected:
                explanation += (
                    "Potential violation: key norm terms are absent from the changed "
                    f"context ({', '.join(sorted(missing_tokens))})."
                )
            else:
                explanation += "No deterministic contradiction signal was found."
            line_start, line_end = _line_range(file.patch)
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
    db.add_all(findings)
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
