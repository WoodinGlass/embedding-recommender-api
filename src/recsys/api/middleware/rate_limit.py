"""HTTP rate-limit middleware (ADR-0014).

Two tiers, in this order:

1. **IP bucket** — always consulted. The limit is
   ``RATE_LIMIT_IP_PER_MINUTE`` (default 5000). The point is to bound
   anonymous or credential-rotating traffic; a caller who rotates
   credentials per request still hits one IP bucket.
2. **Credential bucket** — consulted when a credential is present.
   Its per-class limit is the ``recommend``/``events``/``admin`` value
   from config. The credential is identified by an HMAC-SHA256 hash
   (``rate_limit.keys.credential_bucket_id``); the raw credential never
   reaches Redis or a log line.

The middleware runs before FastAPI routing, so it operates on the raw
path and the request headers. It does not know the parsed body, and it
does not validate the credential; validation is the auth dependency's
job (ADR-0013). The middleware's job is to have *something* to key the
bucket on: a random credential still produces a stable bucket for the
duration of an attack, which is exactly what stops an attacker from
escaping the credential bucket by rotating credentials.

Health, readiness, metrics, and OpenAPI paths are exempt. A rate-limited
``/readyz`` reads to an orchestrator as "not ready" and drains a healthy
instance; the exempt list exists for that reason, not for convenience.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from recsys.monitoring.logging import get_logger
from recsys.monitoring.metrics import (
    RATE_LIMIT_DEGRADED,
    RATE_LIMIT_HITS,
    RATE_LIMIT_REMAINING,
)
from recsys.rate_limit import (
    RateLimitDecision,
    TokenBucketLimiter,
    credential_bucket_id,
    ip_bucket_id,
)

log = get_logger(__name__)

#: Paths that bypass the limiter. ``/docs`` and ``/openapi.json`` are
#: listed because the OpenAPI UI issues a burst of requests on load and
#: a developer behind a NAT should not be locked out of the docs.
_EXEMPT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "/livez",
        "/healthz",
        "/readyz",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

#: Path prefix → rate-limit class. The match is on the request's raw
#: path (before routing), so it is a prefix, not a template. Order
#: matters: the first matching prefix wins, so longer prefixes are
#: listed first. ``/v1/items/`` covers ``/v1/items/{id}/similar``.
_RULES: Final[tuple[tuple[str, str], ...]] = (
    ("/v1/recommend", "recommend"),
    ("/v1/items/", "recommend"),
    ("/v1/events", "events"),
    ("/v1/churn/", "churn"),
)


def classify_path(path: str) -> str | None:
    """Return the rate-limit class for ``path``, or ``None`` when exempt.

    A ``None`` return means "no rate-limit check for this request": the
    path is exempt, or it is not a route the limiter handles. The
    middleware treats ``None`` as "let the request through".
    """
    if path in _EXEMPT_PATHS:
        return None
    for prefix, class_name in _RULES:
        if path.startswith(prefix):
            return class_name
    return None


def extract_raw_credential(request: Request) -> str | None:
    """Return the credential a request carries, or ``None``.

    Checks ``X-API-Key`` first, then ``Authorization: Bearer <token>``.
    Does not validate either; the auth dependency is what validates.
    The purpose here is only to have a stable bucket id: a random
    credential still hashes to a stable bucket for the duration of an
    attack, and that stability is what stops an attacker from escaping
    the credential bucket by rotating credentials.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token or None
    return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Two-tier rate limiting with a per-class credential limit.

    The limiter is shared with any other caller that needs it (the
    cache path in M3.3 will use the same instance). It is passed in,
    not constructed here, so the app factory owns the lifecycle.
    """

    def __init__(
        self,
        app: Any,
        *,
        limiter: TokenBucketLimiter,
        limits: Mapping[str, int],
        ip_limit_per_minute: int,
        trusted_proxy_count: int,
        credential_salt: str,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._limits = dict(limits)
        self._ip_limit = ip_limit_per_minute
        self._trusted_proxy_count = trusted_proxy_count
        self._credential_salt = credential_salt

    # ------------------------------------------------------------------ #
    # middleware
    # ------------------------------------------------------------------ #
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        class_name = classify_path(request.url.path)
        if class_name is None:
            return await call_next(request)

        # --- IP tier (always) ----------------------------------------- #
        ip_id = self._ip_bucket_id(request)
        ip_decision = await self._limiter.check(
            class_name="ip",
            bucket_id=ip_id,
            limit_per_minute=self._ip_limit,
        )
        self._record(class_name="ip", decision=ip_decision)

        if not ip_decision.allowed:
            return self._too_many(
                request,
                decision=ip_decision,
                limit=self._ip_limit,
                class_name="ip",
            )

        # --- Credential tier (only when a credential is present) ------ #
        raw = extract_raw_credential(request)
        credential_id = credential_bucket_id(raw, salt=self._credential_salt)
        if credential_id is None:
            # No credential: the IP tier is the whole check. Attach
            # its headers and let the request through.
            response = await call_next(request)
            self._attach_headers(response, decision=ip_decision, limit=self._ip_limit)
            return response

        limit = self._limits.get(class_name, self._ip_limit)
        credential_decision = await self._limiter.check(
            class_name=class_name,
            bucket_id=credential_id,
            limit_per_minute=limit,
        )
        self._record(class_name=class_name, decision=credential_decision)

        if not credential_decision.allowed:
            return self._too_many(
                request,
                decision=credential_decision,
                limit=limit,
                class_name=class_name,
            )

        response = await call_next(request)
        self._attach_headers(response, decision=credential_decision, limit=limit)
        return response

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _ip_bucket_id(self, request: Request) -> str:
        peer_ip = request.client.host if request.client else None
        forwarded_raw = request.headers.get("X-Forwarded-For", "")
        forwarded = [part.strip() for part in forwarded_raw.split(",") if part.strip()]
        return ip_bucket_id(
            peer_ip=peer_ip,
            forwarded_for=forwarded,
            trusted_proxy_count=self._trusted_proxy_count,
        )

    def _record(self, *, class_name: str, decision: RateLimitDecision) -> None:
        RATE_LIMIT_HITS.labels(
            class_name=class_name,
            result="allowed" if decision.allowed else "limited",
        ).inc()
        RATE_LIMIT_REMAINING.labels(class_name=class_name).observe(max(0, decision.remaining))
        # Surface the degraded state so a Grafana alert can fire on it.
        RATE_LIMIT_DEGRADED.set(1 if decision.degraded else 0)

    def _too_many(
        self,
        request: Request,
        *,
        decision: RateLimitDecision,
        limit: int,
        class_name: str,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        body = {
            "error": {
                "code": "rate_limited",
                "message": (f"Rate limit exceeded for class {class_name!r}."),
                "request_id": request_id,
                "details": {
                    "retry_after_seconds": decision.retry_after_seconds,
                    "limit_class": class_name,
                },
            }
        }
        reset_at = int(time.time()) + max(0, decision.retry_after_seconds)
        headers = {
            "Retry-After": str(decision.retry_after_seconds),
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(decision.remaining),
            "X-RateLimit-Reset": str(reset_at),
        }
        log.info(
            "ratelimit.limited",
            class_name=class_name,
            retry_after_seconds=decision.retry_after_seconds,
            remaining=decision.remaining,
        )
        return JSONResponse(body, status_code=429, headers=headers)

    def _attach_headers(
        self,
        response: Response,
        *,
        decision: RateLimitDecision,
        limit: int,
    ) -> None:
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
