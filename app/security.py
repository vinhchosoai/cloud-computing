"""Authentication, rate limiting, and privacy-safe logging helpers."""

from __future__ import annotations

import hashlib
import hmac

from fastapi import Header, HTTPException, Request, status
from slowapi import Limiter

from app.config import get_settings


def get_rate_limit_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return request.client.host if request.client else "anonymous"


def get_dynamic_rate_limit(key: str) -> str:
    settings = get_settings()
    if settings.benchmark_api_key:
        bench_hash = hashlib.sha256(settings.benchmark_api_key.encode("utf-8")).hexdigest()[:16]
        if hmac.compare_digest(key, bench_hash):
            return settings.benchmark_rate_limit
    return settings.rate_limit


limiter = Limiter(key_func=get_rate_limit_key)


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    valid_keys = get_settings().get_valid_api_keys()
    for valid in valid_keys:
        if hmac.compare_digest(x_api_key, valid):
            return x_api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key",
    )


def hash_text_for_log(text: str) -> str:
    """Return a short non-reversible SHA-256 digest for audit logging without PII."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
