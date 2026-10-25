from __future__ import annotations

import hashlib
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import IngestionJob, Installation, Repository
from app.services.github import GitHubClient
from app.services.ingestion import ingest_repository

REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PUBLIC_SOURCE = "github_public_api"
SYNTHETIC_MARKER = "pr-norms-synthetic-demo-v1"


def parse_repository_identifier(value: str) -> str:
    candidate = value.strip()
    candidate = re.sub(r"^https?://github\.com/", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.removesuffix(".git").strip("/")
    if not REPOSITORY_PATTERN.fullmatch(candidate):
        raise ValueError("use a GitHub URL or owner/repo identifier")
    return candidate


def _installation_id(full_name: str) -> int:
    value = int(hashlib.sha256(full_name.casefold().encode()).hexdigest()[:15], 16)
    return -(value or 1)


async def import_public_repository(
    db: AsyncSession, value: str, max_pull_requests: int = 50
) -> IngestionJob:
    full_name = parse_repository_identifier(value)
    maximum = get_settings().public_import_max_pull_requests
    if not 1 <= max_pull_requests <= maximum:
        raise ValueError(f"max_pull_requests must be between 1 and {maximum}")

    async with GitHubClient(get_settings().github_token or "") as github:
        repository_payload = (await github.get(f"/repos/{full_name}")).json()

    installation = await db.scalar(
        select(Installation).where(
            Installation.github_id == _installation_id(full_name)
        )
    )
    if installation is None:
        installation = Installation(
            github_id=_installation_id(full_name),
            account_login=full_name.split("/", 1)[0],
            account_type="Public",
            raw={"source": PUBLIC_SOURCE},
        )
        db.add(installation)
        await db.flush()

    repository = await db.scalar(
        select(Repository).where(Repository.full_name == full_name)
    )
    if (
        repository is not None
        and repository.raw.get("synthetic_demo") == SYNTHETIC_MARKER
    ):
        raise ValueError("refusing to import over the synthetic demo repository")
    if repository is None:
        repository = Repository(
            installation_id=installation.id,
            github_id=repository_payload["id"],
            full_name=full_name,
            default_branch=repository_payload.get("default_branch") or "main",
            raw={"source": PUBLIC_SOURCE, "github": repository_payload},
        )
        db.add(repository)
        await db.flush()
    else:
        repository.installation_id = installation.id
        repository.github_id = repository_payload["id"]
        repository.default_branch = repository_payload.get("default_branch") or "main"
        repository.raw = {"source": PUBLIC_SOURCE, "github": repository_payload}

    job = IngestionJob(repository_id=repository.id, status="queued")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return await ingest_repository(
        db,
        job.id,
        get_settings().github_token or "",
        max_pull_requests=max_pull_requests,
        recent_closed_only=True,
    )
