"""Phase 3 candidate norm lifecycle and immutable audit history."""

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    CandidateNorm,
    CandidateNormEvidence,
    NormConflict,
    NormVersion,
)

CLASSIFICATIONS = {
    "hard_rule",
    "team_convention",
    "reviewer_preference",
    "one_off",
}
STATUSES = {"emerging", "active", "weakening", "deprecated", "conflicting"}
DECISIONS = {"provisional", "approved", "rejected"}


async def _evidence_count(db: AsyncSession, norm_id: int) -> int:
    return int(
        await db.scalar(
            select(func.count(CandidateNormEvidence.id)).where(
                CandidateNormEvidence.candidate_norm_id == norm_id,
                CandidateNormEvidence.active.is_(True),
            )
        )
        or 0
    )


async def _next_version(db: AsyncSession, norm_id: int) -> int:
    current = await db.scalar(
        select(func.max(NormVersion.version)).where(
            NormVersion.candidate_norm_id == norm_id
        )
    )
    return int(current or 0) + 1


async def record_version(
    db: AsyncSession,
    norm: CandidateNorm,
    *,
    reason: str | None,
    changed_by: str | None,
) -> NormVersion:
    version = NormVersion(
        candidate_norm_id=norm.id,
        version=await _next_version(db, norm.id),
        statement=norm.statement,
        classification=norm.classification,
        confidence=norm.confidence,
        status=norm.status,
        approved=norm.approved,
        decision=norm.decision,
        reason=reason,
        changed_by=changed_by,
        evidence_count=await _evidence_count(db, norm.id),
    )
    db.add(version)
    await db.flush()
    return version


async def ensure_initial_version(db: AsyncSession, norm: CandidateNorm) -> NormVersion:
    existing = await db.scalar(
        select(NormVersion)
        .where(NormVersion.candidate_norm_id == norm.id)
        .order_by(NormVersion.version)
    )
    if existing:
        return existing
    return await record_version(
        db, norm, reason="initial candidate snapshot", changed_by=None
    )


async def update_norm(
    db: AsyncSession,
    norm_id: int,
    *,
    statement: str | None = None,
    classification: str | None = None,
    confidence: float | None = None,
    status: str | None = None,
    reason: str | None = None,
    changed_by: str | None = None,
) -> CandidateNorm:
    norm = await db.get(CandidateNorm, norm_id)
    if norm is None:
        raise ValueError(f"candidate norm {norm_id} not found")
    if classification is not None and classification not in CLASSIFICATIONS:
        raise ValueError(f"invalid classification: {classification}")
    if status is not None and status not in STATUSES:
        raise ValueError(f"invalid status: {status}")
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    evidence_count = await _evidence_count(db, norm.id)
    if status is not None and status != "emerging" and evidence_count == 0:
        raise ValueError("status changes require supporting evidence")
    await ensure_initial_version(db, norm)
    if statement is not None:
        if not statement.strip():
            raise ValueError("statement must not be empty")
        norm.statement = statement
    if classification is not None:
        norm.classification = classification
    if confidence is not None:
        norm.confidence = confidence
    if status is not None:
        norm.status = status
    if changed_by is not None:
        norm.reviewed_by = changed_by
        norm.reviewed_at = datetime.now(UTC).replace(tzinfo=None)
    await record_version(db, norm, reason=reason, changed_by=changed_by)
    await db.commit()
    await db.refresh(norm)
    return norm


async def approve_norm(
    db: AsyncSession, norm_id: int, *, changed_by: str | None, reason: str | None
) -> CandidateNorm:
    norm = await db.get(CandidateNorm, norm_id)
    if norm is None:
        raise ValueError(f"candidate norm {norm_id} not found")
    if await _evidence_count(db, norm_id) == 0:
        raise ValueError("approval requires supporting evidence")
    await ensure_initial_version(db, norm)
    norm.approved = True
    norm.decision = "approved"
    norm.status = "active"
    norm.reviewed_by = changed_by
    norm.reviewed_at = datetime.now(UTC).replace(tzinfo=None)
    await record_version(db, norm, reason=reason, changed_by=changed_by)
    await db.commit()
    await db.refresh(norm)
    return norm


async def reject_norm(
    db: AsyncSession, norm_id: int, *, changed_by: str | None, reason: str | None
) -> CandidateNorm:
    norm = await db.get(CandidateNorm, norm_id)
    if norm is None:
        raise ValueError(f"candidate norm {norm_id} not found")
    await ensure_initial_version(db, norm)
    norm.approved = False
    norm.decision = "rejected"
    norm.status = "deprecated"
    norm.reviewed_by = changed_by
    norm.reviewed_at = datetime.now(UTC).replace(tzinfo=None)
    await record_version(db, norm, reason=reason, changed_by=changed_by)
    await db.commit()
    await db.refresh(norm)
    return norm


async def mark_conflict(
    db: AsyncSession,
    norm_id: int,
    conflicting_norm_id: int,
    *,
    reason: str,
    changed_by: str | None,
) -> None:
    if norm_id == conflicting_norm_id:
        raise ValueError("a norm cannot conflict with itself")
    first, second = sorted((norm_id, conflicting_norm_id))
    if not await db.get(CandidateNorm, first) or not await db.get(
        CandidateNorm, second
    ):
        raise ValueError("both candidate norms must exist")
    first_evidence = await _evidence_count(db, first)
    second_evidence = await _evidence_count(db, second)
    if first_evidence == 0 or second_evidence == 0:
        raise ValueError("conflict status requires supporting evidence for both norms")
    evidence_count = first_evidence + second_evidence
    conflict = await db.scalar(
        select(NormConflict).where(
            NormConflict.candidate_norm_id == first,
            NormConflict.conflicting_norm_id == second,
        )
    )
    if conflict is None:
        conflict = NormConflict(
            candidate_norm_id=first,
            conflicting_norm_id=second,
            reason=reason,
            evidence_count=0,
        )
        db.add(conflict)
    else:
        conflict.reason = reason
    for candidate_id in (first, second):
        norm = await db.get(CandidateNorm, candidate_id)
        if norm is not None:
            await ensure_initial_version(db, norm)
            norm.status = "conflicting"
            await record_version(db, norm, reason=reason, changed_by=changed_by)
    await db.flush()
    conflict.evidence_count = evidence_count
    await db.commit()
