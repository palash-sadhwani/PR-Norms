import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Self

import httpx

from app.core.config import get_settings

# Longer secondary-limit waits are reported instead of blocking the request.
MAX_RETRY_WAIT_SECONDS = 30


class GitHubRequestError(RuntimeError):
    """A sanitized, actionable error from the GitHub REST API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        rate_limited: bool = False,
        reset_at: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.rate_limited = rate_limited
        self.reset_at = reset_at


class GitHubClient:
    def __init__(self, token: str, client: httpx.AsyncClient | None = None):
        # An empty token means anonymous; callers opt in to GITHUB_TOKEN explicitly.
        self.token = token
        self.client = client
        self._owned = client is None

    async def __aenter__(self) -> Self:
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=get_settings().github_api_url)
        return self

    async def __aexit__(self, *_):
        if self._owned and self.client:
            await self.client.aclose()

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        return headers

    def _rate_limit_error(self, response: httpx.Response) -> GitHubRequestError:
        reset_at = response.headers.get("x-ratelimit-reset")
        limit = response.headers.get("x-ratelimit-limit")
        parts = ["GitHub API rate limit exceeded"]
        if limit:
            parts.append(f" ({limit} requests/hour")
            parts.append(", authenticated)" if self.token else ", unauthenticated)")
        parts.append(".")
        if reset_at and reset_at.isdigit():
            when = datetime.fromtimestamp(int(reset_at), UTC).strftime("%H:%M UTC")
            parts.append(f" It resets at {when}.")
        else:
            parts.append(" Try again after the limit resets.")
        if not self.token:
            parts.append(" Set GITHUB_TOKEN for the backend to raise the limit.")
        return GitHubRequestError(
            "".join(parts), status_code=429, rate_limited=True, reset_at=reset_at
        )

    async def get(self, path: str, **params) -> httpx.Response:
        if self.client is None:
            raise RuntimeError("GitHubClient must be used as an async context manager")
        last_response: httpx.Response | None = None
        for attempt in range(4):
            response = await self.client.get(
                path, params=params, headers=self._headers()
            )
            last_response = response
            exhausted = response.headers.get("x-ratelimit-remaining") == "0"
            retry_after = response.headers.get("retry-after")
            if response.status_code in {403, 429} and exhausted:
                # The primary limit only recovers at x-ratelimit-reset; retrying is futile.
                raise self._rate_limit_error(response)
            if (
                response.status_code == 403
                and retry_after is None
                and "rate limit" in response.text.lower()
            ):
                # Secondary limit without retry-after: GitHub asks for a wait of at
                # least a minute, which is reported rather than slept through.
                raise self._rate_limit_error(response)
            secondary = response.status_code in {403, 429} and retry_after is not None
            if secondary or response.status_code in {429, 500, 502, 503, 504}:
                delay = float(retry_after) if retry_after else 2**attempt
                if attempt == 3 or delay > MAX_RETRY_WAIT_SECONDS:
                    break
                await asyncio.sleep(delay)
                continue
            if response.status_code == 404:
                raise GitHubRequestError(
                    "GitHub repository or resource was not found. Private repositories "
                    "also return 404 without access.",
                    status_code=404,
                )
            if response.status_code == 401:
                raise GitHubRequestError(
                    "GitHub rejected the configured token (401). Check GITHUB_TOKEN.",
                    status_code=401,
                )
            if response.status_code >= 400:
                raise GitHubRequestError(
                    f"GitHub API request failed with status {response.status_code}.",
                    status_code=response.status_code,
                )
            return response
        if last_response is not None:
            if last_response.status_code == 429 or (
                last_response.status_code == 403
                and "retry-after" in last_response.headers
            ):
                raise self._rate_limit_error(last_response)
            raise GitHubRequestError(
                "GitHub API request failed after retries.",
                status_code=last_response.status_code,
            )
        raise GitHubRequestError("GitHub API request failed after retries.")

    async def post(self, path: str, **json_body) -> httpx.Response:
        if self.client is None:
            raise RuntimeError("GitHubClient must be used as an async context manager")
        response = await self.client.post(path, json=json_body, headers=self._headers())
        response.raise_for_status()
        return response

    async def graphql(self, query: str, variables: dict) -> dict:
        if self.client is None:
            raise RuntimeError("GitHubClient must be used as an async context manager")
        response = await self.client.post(
            "/graphql",
            json={"query": query, "variables": variables},
            headers=self._headers("application/json"),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"GitHub GraphQL error: {payload['errors']}")
        return payload["data"]

    async def paginate(
        self, path: str, *, items_key: str | None = None, **params
    ) -> AsyncIterator[dict]:
        """Yield every item of a paged list; items_key names a wrapped list."""
        page = 1
        while True:
            response = await self.get(path, page=page, per_page=100, **params)
            items = response.json()
            if items_key is not None:
                items = items.get(items_key, [])
            if not items:
                return
            for item in items:
                yield item
            if len(items) < 100:
                return
            page += 1
