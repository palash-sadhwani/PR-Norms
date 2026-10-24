import hashlib
import hmac

import httpx
import pytest

from app.api.v1 import routes
from app.core.config import get_settings
from app.services.auth import create_oauth_state, verify_oauth_state
from app.services.github import GitHubClient, GitHubRequestError
from app.services.preprocess import preprocess_comment
from app.services.public_import import parse_repository_identifier
from app.services.webhooks import verify_signature


def test_preprocess_preserves_raw_and_normalizes():
    result = preprocess_comment("  hello\n world ")
    assert result.raw_text == "  hello\n world "
    assert result.normalized_text == "hello world"


def test_signature():
    body = b'{"action":"opened"}'
    secret = "secret"
    import os

    os.environ["GITHUB_WEBHOOK_SECRET"] = secret
    get_settings.cache_clear()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_signature(body, signature)
    assert not verify_signature(body, "sha256=bad")


def test_state_is_signed_and_expires(monkeypatch):
    monkeypatch.setattr(get_settings(), "github_state_secret", "state-secret")
    state = create_oauth_state()
    assert verify_oauth_state(state)
    assert not verify_oauth_state(f"{state}tampered")


@pytest.mark.asyncio
async def test_github_pagination_and_retry(monkeypatch):
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"retry-after": "0"})
        page = request.url.params["page"]
        if page == "1":
            return httpx.Response(200, json=[{"id": 1}] * 100)
        return httpx.Response(200, json=[{"id": 2}])

    client = httpx.AsyncClient(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    async with GitHubClient("token", client) as github:
        items = [item async for item in github.paginate("/repos/acme/demo/pulls")]
    assert [item["id"] for item in items] == [1] * 100 + [2]
    assert calls == 3


@pytest.mark.asyncio
async def test_github_pagination_of_wrapped_lists():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"total_count": 2, "repositories": [{"id": 1}, {"id": 2}]}
        )

    client = httpx.AsyncClient(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    async with GitHubClient("token", client) as github:
        items = [
            item
            async for item in github.paginate(
                "/installation/repositories", items_key="repositories"
            )
        ]
    assert [item["id"] for item in items] == [1, 2]


def test_meaningful_feedback_is_not_filtered():
    result = preprocess_comment("LGTM, but move DB access to repository layer")
    assert result.is_low_information is False
    assert result.normalized_text == "LGTM, but move DB access to repository layer"


def test_public_repository_identifier_parsing():
    assert parse_repository_identifier("fastapi/fastapi") == "fastapi/fastapi"
    assert (
        parse_repository_identifier("https://github.com/fastapi/fastapi.git")
        == "fastapi/fastapi"
    )
    with pytest.raises(ValueError):
        parse_repository_identifier("https://example.com/not-github")


def test_public_import_accepts_bounded_counts_above_github_page_size():
    assert (
        routes.PublicImportIn(
            repository="django/django", max_pull_requests=250
        ).max_pull_requests
        == 250
    )
    with pytest.raises(ValueError):
        routes.PublicImportIn(repository="django/django", max_pull_requests=1001)


@pytest.mark.asyncio
async def test_public_client_omits_authentication_header():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"id": 1})

    client = httpx.AsyncClient(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    async with GitHubClient("", client) as github:
        response = await github.get("/repos/acme/demo")
    assert response.json()["id"] == 1


@pytest.mark.asyncio
async def test_rate_limit_is_actionable_and_does_not_become_generic_runtime_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": "1791744000",
                "retry-after": "0",
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    async with GitHubClient("", client) as github:
        with pytest.raises(GitHubRequestError) as error:
            await github.get("/repos/acme/demo")
    assert error.value.rate_limited is True
    assert error.value.status_code == 429
    assert "rate limit" in str(error.value).lower()


@pytest.mark.asyncio
async def test_public_import_route_returns_sanitized_rate_limit_error(monkeypatch):
    async def fail(*args, **kwargs):
        raise GitHubRequestError(
            "GitHub API rate limit exceeded. Try again after the limit resets.",
            status_code=429,
            rate_limited=True,
        )

    monkeypatch.setattr(routes, "import_public_repository", fail)
    with pytest.raises(routes.HTTPException) as error:
        await routes.public_import(
            routes.PublicImportIn(repository="django/django", max_pull_requests=10),
            None,
        )
    assert error.value.status_code == 429
    assert "rate limit" in str(error.value.detail).lower()


@pytest.mark.asyncio
async def test_public_import_route_maps_missing_repository_to_404(monkeypatch):
    async def fail(*args, **kwargs):
        raise GitHubRequestError(
            "GitHub repository or resource was not found.", status_code=404
        )

    monkeypatch.setattr(routes, "import_public_repository", fail)
    with pytest.raises(routes.HTTPException) as error:
        await routes.public_import(
            routes.PublicImportIn(
                repository="missing/repository", max_pull_requests=10
            ),
            None,
        )
    assert error.value.status_code == 404
