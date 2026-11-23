"""Reusable FastAPI dependencies.

* :func:`get_registry` / :func:`get_detection_service` — wiring from ``app.state``
* :func:`require_api_key` — optional static API-key auth (``X-API-Key``)
* :func:`rate_limiter` — optional in-memory per-key budget
* :func:`enforce_query_length` — shared guard for text inputs

Both security features are opt-in: with no ``SQLI_API_KEYS`` set and
``SQLI_RATE_LIMIT_PER_MINUTE=0`` the API is open, which keeps local
development friction low.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import deque

from fastapi import Depends, Header, Request, status

from api.config import Settings, get_settings
from api.errors import APIError, InvalidInputError, PayloadTooLargeError
from api.ml.registry import ModelRegistry
from api.services import DetectionService


class UnauthorizedError(APIError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "unauthorized"


class RateLimitExceededError(APIError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "rate_limit_exceeded"


# ------------------------------------------------------------- wiring


def get_registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover - only if wiring regresses
        raise RuntimeError("Model registry is not attached to app.state")
    return registry


def get_detection_service(
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> DetectionService:
    return DetectionService(registry, settings)


# --------------------------------------------------------------- auth


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> str:
    """Validate ``X-API-Key`` when API keys are configured."""
    if not settings.auth_enabled:
        return "anonymous"

    if not x_api_key:
        raise UnauthorizedError("Missing X-API-Key header.")

    # Constant-time comparison to avoid leaking key prefixes via timing.
    for configured in settings.api_keys:
        if hmac.compare_digest(x_api_key, configured):
            return "keyed"

    raise UnauthorizedError("Invalid API key.")


# ------------------------------------------------------- rate limiting


class SlidingWindowLimiter:
    """Minimal in-memory sliding-window limiter.

    Deliberately simple: correct for a single process, good enough for a
    portfolio/demo deployment. Behind multiple workers or replicas, swap this
    for Redis (see README).
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: float = 60.0) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()
        cutoff = now - window_seconds

        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= limit:
                retry_after = max(0.0, window_seconds - (now - bucket[0]))
                return False, round(retry_after, 2)

            bucket.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter = SlidingWindowLimiter()


def rate_limiter(
    request: Request,
    caller: str = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> str:
    """Apply the configured per-caller request budget."""
    if not settings.rate_limit_enabled:
        return caller

    bucket_key = caller if caller != "anonymous" else (request.client.host if request.client else "unknown")
    allowed, retry_after = _limiter.check(bucket_key, settings.rate_limit_per_minute)

    if not allowed:
        raise RateLimitExceededError(
            f"Rate limit of {settings.rate_limit_per_minute} requests/minute exceeded. "
            f"Retry in {retry_after:.1f}s."
        )
    return caller


def get_limiter() -> SlidingWindowLimiter:
    """Exposed for tests that need to reset state between cases."""
    return _limiter


# ---------------------------------------------------------- validation


def validate_query_length(query: str, settings: Settings) -> str:
    """Guard against oversized single inputs."""
    if len(query) > settings.max_query_length:
        raise InvalidInputError(
            f"query exceeds the maximum length of {settings.max_query_length} characters."
        )
    return query


def enforce_batch_limits(queries: list[str], settings: Settings) -> list[str]:
    """Validate count and per-item length for batch endpoints."""
    if len(queries) > settings.max_batch_size:
        raise PayloadTooLargeError(
            f"Batch contains {len(queries)} queries; the maximum is {settings.max_batch_size}. "
            "Split the request or raise SQLI_MAX_BATCH_SIZE."
        )
    for index, query in enumerate(queries):
        if len(query) > settings.max_query_length:
            raise InvalidInputError(
                f"queries[{index}] exceeds the maximum length of "
                f"{settings.max_query_length} characters."
            )
    return queries
