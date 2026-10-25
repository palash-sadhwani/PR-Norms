import asyncio
import logging
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db
from app.models import (
    CandidateNorm,
    CandidateNormEvidence,
    CommentCluster,
    CommentClusterMember,
    Commit,
    IngestionJob,
    Installation,
    IssueComment,
    NormFinding,
    NormFindingFeedback,
    NormVersion,
    PullRequest,
    PullRequestAnalysis,
    PullRequestFile,
    Repository,
    Review,
    ReviewComment,
)
from app.schemas.common import (
    FindingFeedbackIn,
    IngestionOut,
    NormConflictIn,
    NormDecisionIn,
    NormUpdateIn,
    PublicImportIn,
    PullRequestOut,
    RepositoryOut,
)
from app.services.analysis import RISK_FORMULA, analyze_pull_request, record_feedback
from app.services.auth import (
    app_jwt,
    create_oauth_state,
    installation_token,
    repository_token,
    verify_oauth_state,
)
from app.services.evaluation import evaluate_labels
from app.services.github import GitHubClient, GitHubRequestError
from app.services.ingestion import accept_delivery, ingest_repository
from app.services.norms import approve_norm, mark_conflict, reject_norm, update_norm
from app.services.phase2 import (
    DEMO_FALLBACK_MODEL,
    create_embedder,
    is_fallback_model,
    mine_repository,
)
from app.services.public_import import import_public_repository
from app.services.webhooks import verify_signature
from app.tasks import (
    analyze_pull_request_task,
    ingest_repository_task,
    mine_repository_task,
)

router = APIRouter()
logger = logging.getLogger(__name__)


async def _ensure_worker_available() -> None:
    broker = urlsplit(get_settings().redis_url)
    host = broker.hostname or "localhost"
    port = broker.port or 6379
    writer = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=0.5
        )
    except (OSError, TimeoutError) as exc:
        raise HTTPException(
            503, "Background worker is unavailable. Start Redis and Celery."
        ) from exc
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()


@router.get("/github/install")
async def github_install():
    settings = get_settings()
    if not settings.github_app_slug:
        raise HTTPException(503, "GitHub App slug is not configured")
    state = create_oauth_state()
    response = RedirectResponse(
        f"https://github.com/apps/{settings.github_app_slug}/installations/new?state={state}",
        status_code=303,
    )
    response.set_cookie(
        "github_oauth_state",
        state,
        max_age=600,
        httponly=True,
        secure=settings.github_callback_url.startswith("https://"),
        samesite="lax",
    )
    return response


@router.get("/github/callback")
async def github_callback(
    request: Request,
    installation_id: int | None = None,
    state: str | None = None,
    code: str | None = None,
    setup_action: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    if not state or not verify_oauth_state(state):
        raise HTTPException(400, "invalid or expired OAuth state")
    if request.cookies.get("github_oauth_state") != state:
        raise HTTPException(400, "OAuth state cookie mismatch")
    if not installation_id:
        raise HTTPException(400, "missing installation_id")

    # Installation metadata needs the App JWT; its repositories need the
    # installation token.
    async with GitHubClient(app_jwt()) as github:
        installation = (
            await github.get(f"/app/installations/{installation_id}")
        ).json()
    async with GitHubClient(await installation_token(installation_id)) as github:
        repositories = [
            item
            async for item in github.paginate(
                "/installation/repositories", items_key="repositories"
            )
        ]

    existing = await db.scalar(
        select(Installation).where(Installation.github_id == installation_id)
    )
    if existing:
        existing.account_login = installation["account"]["login"]
        existing.account_type = installation["account"].get("type", "Organization")
        existing.active = True
        installation_row = existing
    else:
        installation_row = Installation(
            github_id=installation_id,
            account_login=installation["account"]["login"],
            account_type=installation["account"].get("type", "Organization"),
        )
        db.add(installation_row)
        await db.flush()
    for item in repositories:
        repository = await db.scalar(
            select(Repository).where(Repository.github_id == item["id"])
        )
        if repository:
            repository.full_name = item["full_name"]
            repository.default_branch = item.get("default_branch") or "main"
            repository.installation_id = installation_row.id
        else:
            db.add(
                Repository(
                    installation_id=installation_row.id,
                    github_id=item["id"],
                    full_name=item["full_name"],
                    default_branch=item.get("default_branch") or "main",
                )
            )
    await db.commit()
    settings = get_settings()
    target = settings.backend_cors_origins.split(",")[0].strip()
    return RedirectResponse(f"{target}/?connected=1", status_code=303)


@router.get("/github/repositories", response_model=list[RepositoryOut])
async def github_repositories(db: AsyncSession = Depends(get_db)):
    return list(
        (await db.scalars(select(Repository).order_by(Repository.full_name))).all()
    )


@router.get("/embedding/config")
async def embedding_config():
    """Which embedder mining and analysis will use, so the UI can label fallbacks."""
    settings = get_settings()
    fallback = settings.phase2_embedding_backend == "demo-hashing"
    return {
        "backend": settings.phase2_embedding_backend,
        "model_name": DEMO_FALLBACK_MODEL if fallback else settings.phase2_embedding_model,
        "fallback": fallback,
    }


@router.get("/connection/status")
async def connection_status():
    return {"status": "ok", "github_configured": bool(get_settings().github_app_id)}


@router.get("/repositories", response_model=list[RepositoryOut])
async def repositories(db: AsyncSession = Depends(get_db)):
    return list(
        (await db.scalars(select(Repository).order_by(Repository.full_name))).all()
    )


@router.post("/repositories/{repository_id}/sync", status_code=202)
async def start_sync(repository_id: int, db: AsyncSession = Depends(get_db)):
    repository = await db.get(Repository, repository_id)
    if not repository:
        raise HTTPException(404, "repository not found")
    job = IngestionJob(repository_id=repository_id, status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    if get_settings().local_demo_mode:
        try:
            token = await repository_token(db, repository)
            result = await ingest_repository(db, job.id, token)
        except (GitHubRequestError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.exception(
                "Local synchronous sync failed for repository %s", repository_id
            )
            raise HTTPException(503, str(exc)) from exc
        return {"job_id": result.id, "status": result.status}
    await _ensure_worker_available()
    try:
        task = ingest_repository_task.apply_async(
            args=[repository_id, job.id], retry=False
        )
    except Exception as exc:
        await db.rollback()
        failed_job = await db.get(IngestionJob, job.id)
        if failed_job is not None:
            failed_job.status = "failed"
            failed_job.error = (
                "Background worker is unavailable. Start Redis and Celery."
            )
            await db.commit()
        logger.exception("Failed to dispatch sync job %s", job.id)
        raise HTTPException(
            503, "Background worker is unavailable. Start Redis and Celery."
        ) from exc
    return {"job_id": job.id, "status": job.status, "task_id": task.id}


@router.get(
    "/repositories/{repository_id}/pull-requests", response_model=list[PullRequestOut]
)
async def pull_requests(repository_id: int, db: AsyncSession = Depends(get_db)):
    return list(
        (
            await db.scalars(
                select(PullRequest).where(PullRequest.repository_id == repository_id)
            )
        ).all()
    )


@router.post("/public/import")
async def public_import(payload: PublicImportIn, db: AsyncSession = Depends(get_db)):
    try:
        job = await import_public_repository(
            db, payload.repository, payload.max_pull_requests, payload.include_open
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except GitHubRequestError as exc:
        logger.exception("Public GitHub import failed for %s", payload.repository)
        if exc.rate_limited:
            raise HTTPException(429, str(exc)) from exc
        if exc.status_code == 404:
            raise HTTPException(404, str(exc)) from exc
        if exc.status_code in (401, 403):
            raise HTTPException(
                403,
                f"GitHub refused access to {payload.repository} ({exc.status_code}). "
                "Private repositories need a GITHUB_TOKEN or GitHub App with read access.",
            ) from exc
        raise HTTPException(503, str(exc)) from exc
    except httpx.HTTPError:
        logger.exception(
            "Public GitHub import request failed for %s", payload.repository
        )
        raise HTTPException(
            503, "GitHub could not be reached. Check the network and try again."
        ) from None
    return {
        "source": "github_public_api",
        "repository_id": job.repository_id,
        "job_id": job.id,
        "status": job.status,
        "total_items": job.total_items,
        "completed_items": job.completed_items,
        "error_count": job.error_count,
        "error": job.error,
    }


@router.get("/pull-requests/{pull_request_id}/reviews")
async def reviews(pull_request_id: int, db: AsyncSession = Depends(get_db)):
    return list(
        (
            await db.scalars(
                select(Review).where(Review.pull_request_id == pull_request_id)
            )
        ).all()
    )


@router.get("/pull-requests/{pull_request_id}/comments")
async def comments(pull_request_id: int, db: AsyncSession = Depends(get_db)):
    return list(
        (
            await db.scalars(
                select(ReviewComment)
                .where(
                    ReviewComment.pull_request_id == pull_request_id,
                    ReviewComment.id.not_in(select(IssueComment.id)),
                )
                .order_by(ReviewComment.id)
            )
        ).all()
    )


@router.get("/pull-requests/{pull_request_id}/issue-comments")
async def issue_comments(pull_request_id: int, db: AsyncSession = Depends(get_db)):
    """General PR conversation, kept apart from line-level review comments."""
    return list(
        (
            await db.scalars(
                select(IssueComment)
                .where(IssueComment.pull_request_id == pull_request_id)
                .order_by(IssueComment.id)
            )
        ).all()
    )


@router.get("/pull-requests/{pull_request_id}/files")
async def files(pull_request_id: int, db: AsyncSession = Depends(get_db)):
    return list(
        (
            await db.scalars(
                select(PullRequestFile).where(
                    PullRequestFile.pull_request_id == pull_request_id
                )
            )
        ).all()
    )


@router.post("/pull-requests/{pull_request_id}/analyze", status_code=202)
async def analyze_pull_request_route(
    pull_request_id: int,
    inline: bool = False,
    db: AsyncSession = Depends(get_db),
):
    if not await db.get(PullRequest, pull_request_id):
        raise HTTPException(404, "pull request not found")
    if inline or get_settings().local_demo_mode:
        try:
            embedder = create_embedder()
            result = await analyze_pull_request(
                db,
                pull_request_id,
                embedder=embedder,
                batch_size=get_settings().phase2_embedding_batch_size,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "status": "completed",
            "embedding_model": embedder.model_name,
            "embedding_fallback": is_fallback_model(embedder.model_name),
            "analysis_id": result.analysis_id,
            "findings": result.findings,
            "context_available": result.context_available,
            "highest_risk": result.highest_risk,
        }
    await _ensure_worker_available()
    try:
        task = analyze_pull_request_task.apply_async(
            args=[pull_request_id], retry=False
        )
    except Exception as exc:
        logger.exception("Failed to dispatch analysis task %s", pull_request_id)
        raise HTTPException(
            503, "Background worker is unavailable. Start Redis and Celery."
        ) from exc
    return {"status": "queued", "task_id": task.id}


@router.get("/pull-requests/{pull_request_id}/analysis")
async def pull_request_analysis(
    pull_request_id: int,
    db: AsyncSession = Depends(get_db),
):
    analysis = await db.scalar(
        select(PullRequestAnalysis).where(
            PullRequestAnalysis.pull_request_id == pull_request_id
        )
    )
    if analysis is None:
        raise HTTPException(404, "analysis not found")
    findings = list(
        (
            await db.scalars(
                select(NormFinding)
                .where(
                    NormFinding.analysis_id == analysis.id,
                    NormFinding.current.is_(True),
                )
                .order_by(NormFinding.risk_score.desc())
            )
        ).all()
    )
    payload = _finding_rows(findings)
    comments = await _comments_by_id(
        db, [i for finding in payload for i in finding["evidence_comment_ids"]]
    )
    pr_numbers = await _pull_request_numbers(
        db, [comment.pull_request_id for comment in comments.values()]
    )
    norm_ids = {finding["candidate_norm_id"] for finding in payload}
    statements = dict(
        (
            await db.execute(
                select(CandidateNorm.id, CandidateNorm.statement).where(
                    CandidateNorm.id.in_(norm_ids)
                )
            )
        ).all()
    ) if norm_ids else {}
    return {
        "id": analysis.id,
        "pull_request_id": analysis.pull_request_id,
        "status": analysis.status,
        "formula_version": analysis.formula_version,
        "context_available": analysis.context_available,
        "findings_count": analysis.findings_count,
        "highest_risk": analysis.highest_risk,
        "scoring_formula": RISK_FORMULA,
        "findings": [
            {
                **finding,
                "norm_statement": statements.get(finding["candidate_norm_id"]),
                "evidence": [
                    {
                        "review_comment_id": comment_id,
                        **_comment_detail(comments.get(comment_id), pr_numbers),
                    }
                    for comment_id in finding["evidence_comment_ids"]
                ],
            }
            for finding in payload
        ],
    }


def _finding_rows(findings: list[NormFinding]) -> list[dict]:
    return [
        {
            "id": finding.id,
            "candidate_norm_id": finding.candidate_norm_id,
            "file_path": finding.file_path,
            "line_start": finding.line_start,
            "line_end": finding.line_end,
            "relevance_score": finding.relevance_score,
            "evidence_strength": finding.evidence_strength,
            "risk_score": finding.risk_score,
            "risk_level": finding.risk_level,
            "provisional": finding.provisional,
            "violation_suspected": finding.violation_suspected,
            "explanation": finding.explanation,
            "evidence_comment_ids": finding.evidence_comment_ids,
            "evidence_pull_request_ids": finding.evidence_pull_request_ids,
            "status": finding.status,
        }
        for finding in findings
    ]


@router.post("/norm-findings/{finding_id}/feedback")
async def finding_feedback(
    finding_id: int,
    payload: FindingFeedbackIn,
    db: AsyncSession = Depends(get_db),
):
    try:
        feedback = await record_feedback(
            db,
            finding_id,
            decision=payload.decision,
            note=payload.note,
            changed_by=payload.changed_by,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return feedback


@router.get("/ingestion/{job_id}", response_model=IngestionOut)
async def ingestion_status(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(IngestionJob, job_id)
    if not job:
        raise HTTPException(404, "ingestion job not found")
    return job


@router.get("/repositories/{repository_id}/stats")
async def repository_stats(repository_id: int, db: AsyncSession = Depends(get_db)):
    pr_ids = select(PullRequest.id).where(PullRequest.repository_id == repository_id)

    async def count(model, *conditions) -> int:
        return int(
            await db.scalar(select(func.count(model.id)).where(*conditions)) or 0
        )

    return {
        "pull_requests": await count(
            PullRequest, PullRequest.repository_id == repository_id
        ),
        "reviews": await count(Review, Review.pull_request_id.in_(pr_ids)),
        "comments": await count(
            ReviewComment,
            ReviewComment.pull_request_id.in_(pr_ids),
            ReviewComment.id.not_in(select(IssueComment.id)),
        ),
        "issue_comments": await count(
            IssueComment, IssueComment.pull_request_id.in_(pr_ids)
        ),
        "files": await count(
            PullRequestFile, PullRequestFile.pull_request_id.in_(pr_ids)
        ),
        "commits": await count(Commit, Commit.pull_request_id.in_(pr_ids)),
        "candidate_norms": await count(
            CandidateNorm, CandidateNorm.repository_id == repository_id
        ),
        "approved_norms": await count(
            CandidateNorm,
            CandidateNorm.repository_id == repository_id,
            CandidateNorm.approved.is_(True),
        ),
        "feedback": await count(
            NormFindingFeedback,
            NormFindingFeedback.finding_id.in_(
                select(NormFinding.id).where(
                    NormFinding.analysis_id.in_(
                        select(PullRequestAnalysis.id).where(
                            PullRequestAnalysis.pull_request_id.in_(pr_ids)
                        )
                    )
                )
            ),
        ),
        "analysis_findings": await count(
            NormFinding,
            NormFinding.current.is_(True),
            NormFinding.analysis_id.in_(
                select(PullRequestAnalysis.id).where(
                    PullRequestAnalysis.pull_request_id.in_(pr_ids)
                )
            ),
        ),
    }


@router.get("/repositories/{repository_id}/evaluation")
async def repository_evaluation(repository_id: int, db: AsyncSession = Depends(get_db)):
    rows = list(
        (
            await db.execute(
                select(
                    NormFinding.id,
                    NormFinding.violation_suspected,
                    NormFindingFeedback.decision,
                )
                .join(
                    NormFindingFeedback,
                    NormFindingFeedback.finding_id == NormFinding.id,
                )
                .join(
                    PullRequestAnalysis,
                    PullRequestAnalysis.id == NormFinding.analysis_id,
                )
                .join(
                    PullRequest, PullRequest.id == PullRequestAnalysis.pull_request_id
                )
                .where(PullRequest.repository_id == repository_id)
                .order_by(NormFinding.id, NormFindingFeedback.id)
            )
        ).all()
    )
    latest: dict[int, tuple[bool, str]] = {
        finding_id: (bool(predicted), decision)
        for finding_id, predicted, decision in rows
    }
    labeled = list(latest.values())
    metrics = evaluate_labels(labeled)
    return {
        "sample_count": metrics.sample_count,
        "labeled_count": metrics.sample_count,
        "positive_label_count": metrics.positive_label_count,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "true_negatives": metrics.true_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
        "undefined_metrics": [
            name
            for name, value in {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "false_positive_rate": metrics.false_positive_rate,
            }.items()
            if value is None
        ],
        "label_definition": "confirmed and accepted are positive; dismissed and needs_review are negative",
    }


@router.get("/repositories/{repository_id}/clusters")
async def semantic_clusters(repository_id: int, db: AsyncSession = Depends(get_db)):
    clusters = list(
        (
            await db.scalars(
                select(CommentCluster)
                .where(
                    CommentCluster.repository_id == repository_id,
                    CommentCluster.active.is_(True),
                )
                .order_by(CommentCluster.label)
            )
        ).all()
    )
    members_by_cluster: dict[int, list[int]] = {cluster.id: [] for cluster in clusters}
    for cluster_id, comment_id in (
        await db.execute(
            select(
                CommentClusterMember.cluster_id, CommentClusterMember.review_comment_id
            ).where(CommentClusterMember.cluster_id.in_(members_by_cluster))
        )
    ).all():
        members_by_cluster[cluster_id].append(comment_id)
    comments_by_id = await _comments_by_id(
        db, [i for ids in members_by_cluster.values() for i in ids]
    )
    return [
        {
            "id": cluster.id,
            "label": cluster.label,
            "model_name": cluster.model_name,
            "algorithm": cluster.algorithm,
            "size": cluster.size,
            "confidence": cluster.confidence,
            "comment_ids": members_by_cluster[cluster.id],
            "comments": [
                {
                    "id": comment.id,
                    "body": comment.body,
                    "path": comment.path,
                    "line": comment.line,
                    "pull_request_id": comment.pull_request_id,
                }
                for comment_id in members_by_cluster[cluster.id]
                if (comment := comments_by_id.get(comment_id))
            ],
        }
        for cluster in clusters
    ]


async def _comments_by_id(
    db: AsyncSession, comment_ids: list[int]
) -> dict[int, ReviewComment]:
    if not comment_ids:
        return {}
    return {
        comment.id: comment
        for comment in (
            await db.scalars(
                select(ReviewComment).where(ReviewComment.id.in_(set(comment_ids)))
            )
        ).all()
    }


async def _pull_request_numbers(db: AsyncSession, ids: list[int]) -> dict[int, int]:
    if not ids:
        return {}
    rows = await db.execute(
        select(PullRequest.id, PullRequest.number).where(PullRequest.id.in_(set(ids)))
    )
    return dict(rows.all())


def _comment_detail(comment: ReviewComment | None, pr_numbers: dict[int, int]) -> dict:
    if comment is None:
        return {"body": None}
    return {
        "body": comment.body,
        "path": comment.path,
        # GitHub nulls `line` once later commits make a comment outdated.
        "line": comment.line if comment.line is not None else comment.original_line,
        "outdated": comment.line is None and comment.original_line is not None,
        "pull_request_number": pr_numbers.get(comment.pull_request_id),
        "html_url": (comment.raw or {}).get("html_url"),
    }


def _evidence_payload(
    evidence: list[CandidateNormEvidence],
    comments: dict[int, ReviewComment],
    pr_numbers: dict[int, int],
) -> list[dict]:
    return [
        {
            "review_comment_id": item.review_comment_id,
            "pull_request_id": item.pull_request_id,
            "active": item.active,
            **_comment_detail(comments.get(item.review_comment_id), pr_numbers),
        }
        for item in evidence
    ]


@router.post("/repositories/{repository_id}/mine", status_code=202)
async def start_semantic_mining(
    repository_id: int,
    model_name: str | None = None,
    inline: bool = False,
    db: AsyncSession = Depends(get_db),
):
    if not await db.get(Repository, repository_id):
        raise HTTPException(404, "repository not found")
    if inline or get_settings().local_demo_mode:
        try:
            embedder = create_embedder(model_name)
            result = await mine_repository(db, repository_id, embedder=embedder)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "status": "completed",
            "embedding_model": embedder.model_name,
            "embedding_fallback": is_fallback_model(embedder.model_name),
            **result.__dict__,
        }
    await _ensure_worker_available()
    try:
        task = mine_repository_task.apply_async(
            args=[repository_id, model_name], retry=False
        )
    except Exception as exc:
        logger.exception("Failed to dispatch mining task %s", repository_id)
        raise HTTPException(
            503, "Background worker is unavailable. Start Redis and Celery."
        ) from exc
    return {"task_id": task.id, "status": "queued"}


@router.get("/repositories/{repository_id}/candidate-norms")
async def candidate_norms(repository_id: int, db: AsyncSession = Depends(get_db)):
    norms = list(
        (
            await db.scalars(
                select(CandidateNorm)
                .where(CandidateNorm.repository_id == repository_id)
                .order_by(CandidateNorm.confidence.desc(), CandidateNorm.id)
            )
        ).all()
    )
    evidence_by_norm: dict[int, list[CandidateNormEvidence]] = {
        norm.id: [] for norm in norms
    }
    for item in (
        await db.scalars(
            select(CandidateNormEvidence)
            .where(CandidateNormEvidence.candidate_norm_id.in_(evidence_by_norm))
            .order_by(CandidateNormEvidence.id)
        )
    ).all():
        evidence_by_norm[item.candidate_norm_id].append(item)
    comments = await _comments_by_id(
        db,
        [item.review_comment_id for rows in evidence_by_norm.values() for item in rows],
    )
    pr_numbers = await _pull_request_numbers(
        db, [comment.pull_request_id for comment in comments.values()]
    )
    return [
        {
            "id": norm.id,
            "cluster_id": norm.cluster_id,
            "statement": norm.statement,
            "confidence": norm.confidence,
            "classification": norm.classification,
            "status": norm.status,
            "approved": norm.approved,
            "decision": norm.decision,
            "support_count": norm.support_count,
            "evidence": _evidence_payload(evidence_by_norm[norm.id], comments, pr_numbers),
        }
        for norm in norms
    ]


@router.get("/candidate-norms/{norm_id}")
async def candidate_norm_detail(norm_id: int, db: AsyncSession = Depends(get_db)):
    norm = await db.get(CandidateNorm, norm_id)
    if norm is None:
        raise HTTPException(404, "candidate norm not found")
    evidence = list(
        (
            await db.scalars(
                select(CandidateNormEvidence)
                .where(CandidateNormEvidence.candidate_norm_id == norm_id)
                .order_by(CandidateNormEvidence.active.desc(), CandidateNormEvidence.id)
            )
        ).all()
    )
    comments = await _comments_by_id(db, [item.review_comment_id for item in evidence])
    pr_numbers = await _pull_request_numbers(
        db, [comment.pull_request_id for comment in comments.values()]
    )
    history = list(
        (
            await db.scalars(
                select(NormVersion)
                .where(NormVersion.candidate_norm_id == norm_id)
                .order_by(NormVersion.version)
            )
        ).all()
    )
    return {
        "id": norm.id,
        "repository_id": norm.repository_id,
        "cluster_id": norm.cluster_id,
        "statement": norm.statement,
        "classification": norm.classification,
        "confidence": norm.confidence,
        "status": norm.status,
        "approved": norm.approved,
        "decision": norm.decision,
        "support_count": norm.support_count,
        "evidence": _evidence_payload(evidence, comments, pr_numbers),
        "history": [
            {
                "version": item.version,
                "statement": item.statement,
                "classification": item.classification,
                "confidence": item.confidence,
                "status": item.status,
                "approved": item.approved,
                "decision": item.decision,
                "reason": item.reason,
                "changed_by": item.changed_by,
                "evidence_count": item.evidence_count,
                "created_at": item.created_at,
            }
            for item in history
        ],
    }


@router.patch("/candidate-norms/{norm_id}")
async def update_candidate_norm(
    norm_id: int,
    payload: NormUpdateIn,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await update_norm(db, norm_id, **payload.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/candidate-norms/{norm_id}/classify")
async def classify_candidate_norm(
    norm_id: int,
    payload: NormUpdateIn,
    db: AsyncSession = Depends(get_db),
):
    if payload.classification is None:
        raise HTTPException(400, "classification is required")
    try:
        return await update_norm(
            db,
            norm_id,
            classification=payload.classification,
            reason=payload.reason,
            changed_by=payload.changed_by,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/candidate-norms/{norm_id}/approve")
async def approve_candidate_norm(
    norm_id: int,
    payload: NormDecisionIn,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await approve_norm(
            db,
            norm_id,
            changed_by=payload.changed_by,
            reason=payload.reason,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/candidate-norms/{norm_id}/reject")
async def reject_candidate_norm(
    norm_id: int,
    payload: NormDecisionIn,
    db: AsyncSession = Depends(get_db),
):
    try:
        return await reject_norm(
            db,
            norm_id,
            changed_by=payload.changed_by,
            reason=payload.reason,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/candidate-norms/{norm_id}/conflicts")
async def conflict_candidate_norm(
    norm_id: int,
    payload: NormConflictIn,
    db: AsyncSession = Depends(get_db),
):
    try:
        await mark_conflict(
            db,
            norm_id,
            payload.conflicting_norm_id,
            reason=payload.reason,
            changed_by=payload.changed_by,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"status": "conflicting"}


@router.get("/repositories/{repository_id}/data-quality")
async def data_quality(repository_id: int, db: AsyncSession = Depends(get_db)):
    prs = list(
        (
            await db.scalars(
                select(PullRequest).where(PullRequest.repository_id == repository_id)
            )
        ).all()
    )
    return {
        "repository_id": repository_id,
        "pull_requests": len(prs),
        "missing_bodies": sum(p.body is None for p in prs),
        "failed_imports": len(
            (
                await db.scalars(
                    select(IngestionJob).where(
                        IngestionJob.repository_id == repository_id,
                        IngestionJob.status == "failed",
                    )
                )
            ).all()
        ),
    }


@router.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(None),
    x_github_delivery: str | None = Header(None),
    x_github_event: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
):
    body = await request.body()
    if not verify_signature(body, x_hub_signature_256):
        raise HTTPException(401, "invalid signature")
    if not x_github_delivery:
        raise HTTPException(400, "missing delivery id")
    payload = await request.json()
    try:
        accepted = await accept_delivery(
            db, x_github_delivery, x_github_event or "unknown", payload
        )
    except IntegrityError:
        await db.rollback()
        accepted = False
    if not accepted:
        return {"status": "duplicate"}
    repository = None
    repo_payload = payload.get("repository") or {}
    if repo_payload.get("id"):
        repository = await db.scalar(
            select(Repository).where(Repository.github_id == repo_payload["id"])
        )
    if repository and x_github_event in {
        "pull_request",
        "pull_request_review",
        "pull_request_review_comment",
    }:
        pr_payload = payload.get("pull_request") or {}
        job = IngestionJob(
            repository_id=repository.id,
            status="queued",
            target_pr_number=pr_payload.get("number"),
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        try:
            ingest_repository_task.apply_async(
                args=[repository.id, job.id, job.target_pr_number], retry=False
            )
        except Exception as exc:
            logger.exception("Failed to dispatch webhook sync job %s", job.id)
            job.status = "failed"
            job.error = "Background worker is unavailable. Start Redis and Celery."
            await db.commit()
            raise HTTPException(
                503, "Background worker is unavailable. Start Redis and Celery."
            ) from exc
        return {"status": "accepted", "job_id": job.id}
    return {"status": "accepted"}
