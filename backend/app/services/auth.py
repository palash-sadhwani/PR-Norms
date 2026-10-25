import base64
import hashlib
import hmac
import json
import secrets
import time

import httpx
import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import Installation, Repository


def _state_secret() -> bytes:
    settings = get_settings()
    secret = settings.github_state_secret or settings.github_webhook_secret
    if not secret:
        raise RuntimeError("GitHub state secret is not configured")
    return secret.encode()


def create_oauth_state() -> str:
    payload = {"nonce": secrets.token_urlsafe(24), "exp": int(time.time()) + 600}
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
    signature = hmac.new(_state_secret(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def verify_oauth_state(state: str) -> bool:
    try:
        encoded, signature = state.split(".", 1)
        expected = hmac.new(
            _state_secret(), encoded.encode(), hashlib.sha256
        ).hexdigest()
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode()))
        return hmac.compare_digest(signature, expected) and int(payload["exp"]) >= int(
            time.time()
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def app_jwt() -> str:
    settings = get_settings()
    if not settings.github_app_id or not settings.github_private_key:
        raise RuntimeError("GitHub App credentials are not configured")
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 540, "iss": str(settings.github_app_id)}
    # .env files usually hold the PEM on one line with escaped newlines.
    private_key = settings.github_private_key.replace("\\n", "\n")
    return jwt.encode(payload, private_key, algorithm="RS256")


async def repository_token(db: AsyncSession, repository: Repository) -> str:
    """Token for ingesting a repository; public imports use the optional GITHUB_TOKEN."""
    if repository.source == "github_public_api":
        return get_settings().github_token or ""
    installation = await db.get(Installation, repository.installation_id)
    if installation is None:
        raise ValueError(f"installation {repository.installation_id} not found")
    return await installation_token(installation.github_id)


async def installation_token(installation_id: int) -> str:
    settings = get_settings()
    async with httpx.AsyncClient(base_url=settings.github_api_url) as client:
        response = await client.post(
            f"/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        response.raise_for_status()
        return response.json()["token"]


async def exchange_oauth_code(code: str) -> dict:
    settings = get_settings()
    if not settings.github_client_id or not settings.github_client_secret:
        raise RuntimeError("GitHub OAuth credentials are not configured")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
                "redirect_uri": settings.github_callback_url,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"GitHub OAuth exchange failed: {data['error']}")
        return data
