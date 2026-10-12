import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models import (
    Commit,
    IngestionJob,
    Installation,
    IssueComment,
    PullRequest,
    PullRequestFile,
    Repository,
    Review,
    ReviewComment,
    ReviewThread,
)
from app.services import ingestion


class FakeResponse:
    def __init__(self, payload: dict | list):
        self.payload = payload

    def json(self):
        return self.payload


class FakeGitHub:
    def __init__(self, _token):
        self.responses = {
            "/repos/acme/demo/pulls/1": {
                "id": 101,
                "number": 1,
                "title": "Improve persistence",
                "body": "Details",
                "state": "closed",
                "user": {"id": 7, "login": "author"},
                "base": {"ref": "main", "sha": "base"},
                "head": {"ref": "feature", "sha": "head"},
                "created_at": "2024-01-01T00:00:00Z",
            }
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, path, **_params):
        return FakeResponse(self.responses[path])

    async def paginate(self, path, **_params):
        values = {
            "/repos/acme/demo/pulls/1/reviews": [
                {
                    "id": 201,
                    "user": {"id": 8, "login": "reviewer"},
                    "state": "COMMENTED",
                    "body": "Please use the repository layer.",
                    "submitted_at": "2024-01-01T01:00:00Z",
                }
            ],
            "/repos/acme/demo/pulls/1/comments": [
                {
                    "id": 301,
                    "pull_request_review_id": 201,
                    "user": {"id": 8, "login": "reviewer"},
                    "body": "Move this query to the repository layer.",
                    "path": "app/db.py",
                    "line": 12,
                    "commit_id": "abc",
                    "diff_hunk": "@@ -1 +1 @@",
                }
            ],
            "/repos/acme/demo/issues/1/comments": [
                {
                    "id": 401,
                    "user": {"id": 9, "login": "commenter"},
                    "body": "Thanks for the fix!",
                }
            ],
            "/repos/acme/demo/pulls/1/files": [
                {
                    "filename": "app/db.py",
                    "status": "modified",
                    "additions": 2,
                    "deletions": 1,
                    "patch": "@@ -1 +1 @@",
                }
            ],
            "/repos/acme/demo/pulls/1/commits": [
                {
                    "sha": "abc",
                    "commit": {"message": "Improve persistence"},
                }
            ],
        }
        for value in values.get(path, []):
            yield value

    async def graphql(self, _query, _variables):
        return {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [
                            {
                                "id": "thread-1",
                                "isResolved": True,
                                "isOutdated": False,
                                "path": "app/db.py",
                                "line": 12,
                                "comments": {"nodes": [{"databaseId": 301}]},
                            }
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }


class FakeRecentGitHub:
    def __init__(self, _token):
        self.pull_requests = [
            {
                "id": 101 + index,
                "number": 30 - index,
                "title": f"Recent change {index}",
                "body": None,
                "state": "closed",
                "user": {"id": 7, "login": "author"},
                "base": {"ref": "main", "sha": "base"},
                "head": {"ref": f"feature-{index}", "sha": f"head-{index}"},
                "created_at": "2024-01-01T00:00:00Z",
            }
            for index in range(3)
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def paginate(self, path, **_params):
        if path == "/repos/acme/demo/pulls":
            for item in self.pull_requests:
                yield item
            return
        return
        yield

    async def graphql(self, _query, _variables):
        raise RuntimeError("GraphQL unavailable")


@pytest.mark.asyncio
async def test_ingestion_materializes_all_phase_one_records_and_is_idempotent(
    monkeypatch,
):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(ingestion, "GitHubClient", FakeGitHub)
    async with session_factory() as db:
        installation = Installation(github_id=1, account_login="acme")
        db.add(installation)
        await db.flush()
        repository = Repository(
            installation_id=installation.id,
            github_id=2,
            full_name="acme/demo",
        )
        db.add(repository)
        await db.flush()
        job = IngestionJob(repository_id=repository.id)
        db.add(job)
        await db.commit()

        result = await ingestion.ingest_repository(
            db, job.id, "token", target_pr_number=1
        )
        assert result.error is None, result.error
        result = await ingestion.ingest_repository(
            db, job.id, "token", target_pr_number=1
        )
        assert result.error is None, result.error

        assert await db.scalar(select(func.count()).select_from(PullRequest)) == 1
        assert await db.scalar(select(func.count()).select_from(Review)) == 1
        assert await db.scalar(select(func.count()).select_from(ReviewComment)) == 2
        assert await db.scalar(select(func.count()).select_from(IssueComment)) == 1
        assert await db.scalar(select(func.count()).select_from(PullRequestFile)) == 1
        assert await db.scalar(select(func.count()).select_from(Commit)) == 1
        assert await db.scalar(select(func.count()).select_from(ReviewThread)) == 1

        comment = await db.scalar(select(ReviewComment))
        thread = await db.scalar(select(ReviewThread))
        assert comment.thread_id == thread.id
        assert comment.raw["id"] == 301
        assert comment.normalized_body == "Move this query to the repository layer."
        assert thread.is_resolved is True

    await engine.dispose()


@pytest.mark.asyncio
async def test_recent_import_processes_bound_and_uses_descending_checkpoint(
    monkeypatch,
):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(ingestion, "GitHubClient", FakeRecentGitHub)
    async with session_factory() as db:
        installation = Installation(github_id=11, account_login="acme")
        db.add(installation)
        await db.flush()
        repository = Repository(
            installation_id=installation.id,
            github_id=12,
            full_name="acme/demo",
        )
        db.add(repository)
        await db.flush()
        first_job = IngestionJob(repository_id=repository.id)
        db.add(first_job)
        await db.commit()

        result = await ingestion.ingest_repository(
            db,
            first_job.id,
            "",
            max_pull_requests=2,
            recent_closed_only=True,
        )
        assert result.status == "completed"
        assert result.total_items == 2
        assert result.completed_items == 2
        assert await db.scalar(select(func.count()).select_from(PullRequest)) == 2

        second_job = IngestionJob(repository_id=repository.id)
        db.add(second_job)
        await db.commit()
        result = await ingestion.ingest_repository(
            db,
            second_job.id,
            "",
            max_pull_requests=2,
            recent_closed_only=True,
        )
        assert result.status == "completed"
        assert result.completed_items == 2
        assert await db.scalar(select(func.count()).select_from(PullRequest)) == 2

    await engine.dispose()
