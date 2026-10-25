import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.base import Base
from app.models import IngestionJob, Installation, PullRequest, Repository
from app.services import ingestion
from app.services.github import GitHubRequestError


def pull(number: int) -> dict:
    return {
        "id": 1000 + number,
        "number": number,
        "title": f"Change {number}",
        "state": "closed",
        "user": {"id": 7, "login": "author"},
        "base": {"ref": "main", "sha": "base"},
        "head": {"ref": f"feature-{number}", "sha": f"head-{number}"},
        "created_at": "2024-01-01T00:00:00Z",
    }


class FailingGitHub:
    """Four PRs; fetching PR #2's reviews fails with `failure`."""

    failure: Exception = GitHubRequestError("GitHub API request failed after retries.")
    requested: list[str]

    def __init__(self, _token):
        FailingGitHub.requested = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def paginate(self, path, **_params):
        self.requested.append(path)
        if path == "/repos/acme/demo/pulls":
            for number in (1, 2, 3, 4):
                yield pull(number)
            return
        if path == "/repos/acme/demo/pulls/2/reviews":
            raise self.failure
        return
        yield


async def setup(db) -> int:
    installation = Installation(github_id=1, account_login="acme")
    db.add(installation)
    await db.flush()
    repository = Repository(installation_id=installation.id, github_id=2, full_name="acme/demo")
    db.add(repository)
    await db.flush()
    job = IngestionJob(repository_id=repository.id)
    db.add(job)
    await db.commit()
    return job.id


@pytest.fixture
async def session(monkeypatch):
    monkeypatch.setattr(ingestion, "GitHubClient", FailingGitHub)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        yield db
    await engine.dispose()


@pytest.mark.asyncio
async def test_one_failed_pull_request_does_not_break_the_rest(session):
    FailingGitHub.failure = GitHubRequestError(
        "GitHub API request failed after retries.", status_code=502
    )
    job_id = await setup(session)

    job = await ingestion.ingest_repository(
        session, job_id, "", max_pull_requests=4, recent_closed_only=True
    )

    # Before the fix, the rollback expired `repository` and every later PR
    # failed with MissingGreenlet ("greenlet_spawn has not been called").
    assert "greenlet" not in (job.error or "")
    assert job.status == "partial_failed"
    assert (job.total_items, job.completed_items, job.error_count) == (4, 3, 1)
    numbers = (await session.scalars(select(PullRequest.number))).all()
    assert sorted(numbers) == [1, 3, 4]


@pytest.mark.asyncio
async def test_rate_limit_stops_the_import_and_reports_it(session):
    FailingGitHub.failure = GitHubRequestError(
        "GitHub API rate limit exceeded.",
        status_code=429,
        rate_limited=True,
        reset_at="1791744000",
    )
    job_id = await setup(session)

    job = await ingestion.ingest_repository(
        session, job_id, "", max_pull_requests=4, recent_closed_only=True
    )

    assert job.status == "rate_limited"
    assert (job.total_items, job.completed_items, job.error_count) == (2, 1, 1)
    assert "rate limit" in job.error.lower()
    # No further GitHub requests once the limit is hit.
    assert not any("/pulls/3/" in path or "/pulls/4/" in path for path in FailingGitHub.requested)
    assert await session.scalar(select(func.count()).select_from(PullRequest)) == 1


@pytest.mark.asyncio
async def test_retry_after_partial_import_does_not_duplicate(session):
    FailingGitHub.failure = GitHubRequestError("boom", status_code=502)
    job_id = await setup(session)
    await ingestion.ingest_repository(
        session, job_id, "", max_pull_requests=4, recent_closed_only=True
    )
    FailingGitHub.failure = None  # PR #2 now succeeds

    class Healthy(FailingGitHub):
        async def paginate(self, path, **params):
            if path == "/repos/acme/demo/pulls/2/reviews":
                return
            async for item in super().paginate(path, **params):
                yield item

    ingestion.GitHubClient = Healthy
    retry = IngestionJob(repository_id=(await session.get(IngestionJob, job_id)).repository_id)
    session.add(retry)
    await session.commit()
    job = await ingestion.ingest_repository(
        session, retry.id, "", max_pull_requests=4, recent_closed_only=True
    )

    assert job.status == "completed"
    numbers = (await session.scalars(select(PullRequest.number))).all()
    assert sorted(numbers) == [1, 2, 3, 4]


def _client(handler, token=""):
    import httpx

    from app.services.github import GitHubClient

    transport = httpx.MockTransport(handler)
    return GitHubClient(token, httpx.AsyncClient(base_url="https://api.github.com", transport=transport))


@pytest.mark.asyncio
async def test_exhausted_primary_limit_fails_fast_with_reset_time(monkeypatch):
    import httpx

    calls = []
    sleeps = []

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("app.services.github.asyncio.sleep", no_sleep)

    def handler(request):
        calls.append(request)
        return httpx.Response(
            403,
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-limit": "60",
                "x-ratelimit-reset": "1791744000",
            },
        )

    async with _client(handler) as github:
        with pytest.raises(GitHubRequestError) as error:
            await github.get("/repos/acme/demo")
    assert len(calls) == 1 and sleeps == []
    message = str(error.value)
    assert error.value.rate_limited and error.value.status_code == 429
    assert "60 requests/hour, unauthenticated" in message
    assert "UTC" in message and "GITHUB_TOKEN" in message


@pytest.mark.asyncio
async def test_authenticated_rate_limit_message_never_contains_the_token():
    import httpx

    def handler(request):
        return httpx.Response(
            429, headers={"x-ratelimit-remaining": "0", "x-ratelimit-limit": "5000"}
        )

    async with _client(handler, token="ghp_secret_value") as github:
        with pytest.raises(GitHubRequestError) as error:
            await github.get("/repos/acme/demo")
    assert "authenticated" in str(error.value)
    assert "ghp_secret_value" not in str(error.value)


@pytest.mark.asyncio
async def test_secondary_limit_waits_retry_after_then_succeeds(monkeypatch):
    import httpx

    sleeps = []

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("app.services.github.asyncio.sleep", no_sleep)
    responses = iter(
        [httpx.Response(403, headers={"retry-after": "2"}), httpx.Response(200, json={"id": 1})]
    )

    async with _client(lambda request: next(responses)) as github:
        response = await github.get("/repos/acme/demo")
    assert response.json() == {"id": 1} and sleeps == [2.0]


@pytest.mark.asyncio
async def test_long_retry_after_is_reported_instead_of_blocking(monkeypatch):
    import httpx

    sleeps = []

    async def no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("app.services.github.asyncio.sleep", no_sleep)

    async with _client(lambda request: httpx.Response(403, headers={"retry-after": "600"})) as github:
        with pytest.raises(GitHubRequestError) as error:
            await github.get("/repos/acme/demo")
    assert error.value.rate_limited and sleeps == []


@pytest.mark.asyncio
async def test_bad_token_is_reported_as_401():
    import httpx

    async with _client(lambda request: httpx.Response(401), token="bad") as github:
        with pytest.raises(GitHubRequestError) as error:
            await github.get("/repos/acme/demo")
    assert error.value.status_code == 401 and "GITHUB_TOKEN" in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,status",
    [
        (GitHubRequestError("GitHub rejected the configured token (401).", status_code=401), 401),
        (GitHubRequestError("not found", status_code=404), 404),
        (GitHubRequestError("forbidden", status_code=403), 403),
    ],
)
async def test_public_import_route_maps_github_errors(monkeypatch, failure, status):
    from app.api.v1 import routes

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(routes, "import_public_repository", fail)
    with pytest.raises(routes.HTTPException) as error:
        await routes.public_import(routes.PublicImportIn(repository="acme/demo"), None)
    assert error.value.status_code == status


@pytest.mark.asyncio
async def test_public_import_route_reports_database_failures_separately(monkeypatch):
    from sqlalchemy.exc import OperationalError

    from app.api.v1 import routes

    async def fail(*args, **kwargs):
        raise OperationalError("INSERT", {}, Exception("disk I/O error"))

    monkeypatch.setattr(routes, "import_public_repository", fail)
    with pytest.raises(routes.HTTPException) as error:
        await routes.public_import(routes.PublicImportIn(repository="acme/demo"), None)
    assert error.value.status_code == 500
    assert "database" in error.value.detail.lower()


@pytest.mark.asyncio
async def test_connection_status_reports_token_presence_not_value(monkeypatch):
    from app.api.v1 import routes
    from app.core.config import get_settings

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret_value")
    get_settings.cache_clear()
    try:
        status = await routes.connection_status()
    finally:
        get_settings.cache_clear()
    assert status["github_token_configured"] is True
    assert "ghp_secret_value" not in str(status)


@pytest.mark.asyncio
async def test_secondary_limit_without_retry_after_is_reported_as_rate_limit():
    import httpx

    body = {"message": "You have exceeded a secondary rate limit."}
    async with _client(lambda request: httpx.Response(403, json=body)) as github:
        with pytest.raises(GitHubRequestError) as error:
            await github.get("/repos/acme/demo")
    assert error.value.rate_limited


@pytest.mark.asyncio
async def test_permission_403_is_not_mistaken_for_a_rate_limit():
    import httpx

    body = {"message": "Resource not accessible by integration"}
    async with _client(lambda request: httpx.Response(403, json=body)) as github:
        with pytest.raises(GitHubRequestError) as error:
            await github.get("/repos/acme/demo")
    assert not error.value.rate_limited and error.value.status_code == 403
