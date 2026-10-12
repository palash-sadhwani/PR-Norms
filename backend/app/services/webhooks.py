import hashlib
import hmac

from app.core.config import get_settings


def verify_signature(body: bytes, signature: str | None) -> bool:
    secret = get_settings().github_webhook_secret
    if not secret or not signature or not signature.startswith("sha256="):
        return False
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature[7:], digest)
