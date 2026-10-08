"""Unit tests for the rate-limit middleware: path classification,
credential extraction, and dispatch (with a fake limiter, no Redis)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request as StarletteRequest
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from recsys.api.middleware.rate_limit import (
    RateLimitMiddleware,
    classify_path,
    extract_raw_credential,
)
from recsys.rate_limit.limiter import RateLimitDecision


# ---------------------------------------------------------------- #
# classify_path
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    ["/livez", "/healthz", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"],
)
def test_classify_path_exempts_health_and_docs(path: str) -> None:
    assert classify_path(path) is None


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/v1/recommend", "recommend"),
        ("/v1/items/i_1/similar", "recommend"),
        ("/v1/events", "events"),
        ("/v1/churn/score", "churn"),
    ],
)
def test_classify_path_matches_known_prefixes(path: str, expected: str) -> None:
    assert classify_path(path) == expected


@pytest.mark.parametrize("path", ["/", "/v1/unknown", "/v2/recommend"])
def test_classify_path_returns_none_for_unmatched(path: str) -> None:
    assert classify_path(path) is None


# ---------------------------------------------------------------- #
# extract_raw_credential
# ---------------------------------------------------------------- #
def _make_request(headers: list[tuple[bytes, bytes]]) -> StarletteRequest:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/recommend",
        "headers": headers,
        "query_string": b"",
    }
    return StarletteRequest(scope)


def test_extract_prefers_x_api_key() -> None:
    req = _make_request(
        [
            (b"x-api-key", b"api-key-value"),
            (b"authorization", b"Bearer jwt-token"),
        ]
    )
    assert extract_raw_credential(req) == "api-key-value"


def test_extract_bearer() -> None:
    req = _make_request([(b"authorization", b"Bearer jwt-token")])
    assert extract_raw_credential(req) == "jwt-token"


def test_extract_returns_none_when_absent() -> None:
    req = _make_request([])
    assert extract_raw_credential(req) is None


def test_extract_empty_bearer_returns_none() -> None:
    req = _make_request([(b"authorization", b"Bearer   ")])
    assert extract_raw_credential(req) is None


def test_extract_non_bearer_authorization_returns_none() -> None:
    req = _make_request([(b"authorization", b"Basic dXNlcjpwYXNz")])
    assert extract_raw_credential(req) is None


# ---------------------------------------------------------------- #
# middleware dispatch (httpx ASGITransport)
# ---------------------------------------------------------------- #
def test_exempt_path_skips_limiter() -> None:
    fake = _FakeLimiter([_allow()])
    app = _make_app(fake, extra_route=("/healthz", _healthz))
    r = _call(app, "GET", "/healthz")
    assert r.status_code == 200
    assert fake.calls == []


def test_ip_and_credential_allowed_returns_200_with_headers() -> None:
    fake = _FakeLimiter([_allow(remaining=42), _allow(remaining=7)])
    app = _make_app(fake)
    r = _call(app, "POST", "/v1/recommend", headers={"X-API-Key": "k"})
    assert r.status_code == 200
    assert r.headers["X-RateLimit-Remaining"] == "7"
    assert r.headers["X-RateLimit-Limit"] == "600"
    assert [c[0] for c in fake.calls] == ["ip", "recommend"]


def test_ip_denied_returns_429_with_retry_after() -> None:
    fake = _FakeLimiter([_deny(retry_after=17)])
    app = _make_app(fake)
    r = _call(app, "POST", "/v1/recommend")
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "17"
    assert r.headers["X-RateLimit-Limit"] == "5000"
    reset = int(r.headers["X-RateLimit-Reset"])
    assert reset > 1_700_000_000, "X-RateLimit-Reset should be an epoch second"
    body = r.json()
    assert body["error"]["code"] == "rate_limited"
    assert body["error"]["details"]["limit_class"] == "ip"


def test_credential_denied_returns_429() -> None:
    fake = _FakeLimiter([_allow(), _deny(retry_after=3, remaining=0)])
    app = _make_app(fake)
    r = _call(app, "POST", "/v1/recommend", headers={"X-API-Key": "k"})
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "3"
    assert r.headers["X-RateLimit-Limit"] == "600"
    assert [c[0] for c in fake.calls] == ["ip", "recommend"]


def test_no_credential_only_ip_tier_runs() -> None:
    fake = _FakeLimiter([_allow(remaining=9)])
    app = _make_app(fake)
    r = _call(app, "POST", "/v1/recommend")
    assert r.status_code == 200
    assert [c[0] for c in fake.calls] == ["ip"]
    assert r.headers["X-RateLimit-Remaining"] == "9"


def test_ip_bucket_id_uses_forwarded_when_trusted() -> None:
    fake = _FakeLimiter([_allow()])
    app = _make_app(fake, trusted_proxy_count=1)
    _call(app, "POST", "/v1/recommend", headers={"X-Forwarded-For": "198.51.100.7"})
    assert fake.calls[0][1] == "198.51.100.7"


def test_ip_bucket_id_falls_back_to_peer_when_trusted_zero() -> None:
    fake = _FakeLimiter([_allow()])
    app = _make_app(fake, trusted_proxy_count=0)
    _call(app, "POST", "/v1/recommend", headers={"X-Forwarded-For": "198.51.100.7"})
    # peer from ASGITransport is deterministic (non-empty); assert it is not the header
    assert fake.calls[0][1] != "198.51.100.7"


# ---------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------- #
async def _healthz(request: StarletteRequest) -> PlainTextResponse:
    return PlainTextResponse("ok")


async def _ok(request: StarletteRequest) -> PlainTextResponse:
    return PlainTextResponse("ok")


def _allow(*, remaining: int = 1) -> RateLimitDecision:
    return RateLimitDecision(
        allowed=True, remaining=remaining, retry_after_seconds=0, degraded=False
    )


def _deny(*, retry_after: int = 1, remaining: int = 0) -> RateLimitDecision:
    return RateLimitDecision(
        allowed=False,
        remaining=remaining,
        retry_after_seconds=retry_after,
        degraded=False,
    )


class _FakeLimiter:
    def __init__(self, decisions: Iterable[RateLimitDecision]) -> None:
        self._decisions = list(decisions)
        self.calls: list[tuple[str, str, int]] = []

    async def check(
        self, *, class_name: str, bucket_id: str, limit_per_minute: int
    ) -> RateLimitDecision:
        self.calls.append((class_name, bucket_id, limit_per_minute))
        if not self._decisions:
            raise AssertionError("FakeLimiter exhausted")
        return self._decisions.pop(0)


def _make_app(
    fake: _FakeLimiter,
    *,
    limits: dict[str, int] | None = None,
    ip_limit: int = 5000,
    trusted_proxy_count: int = 0,
    salt: str = "test-salt",
    extra_route: tuple[str, Any] | None = None,
) -> Starlette:
    routes: list[Route] = [Route("/v1/recommend", _ok, methods=["POST"])]
    if extra_route is not None:
        path, handler = extra_route
        routes.append(Route(path, handler, methods=["GET"]))
    app = Starlette(routes=routes)
    app.add_middleware(
        RateLimitMiddleware,
        limiter=fake,  # type: ignore[arg-type]
        limits=limits or {"recommend": 600, "events": 6000, "admin": 60, "churn": 600},
        ip_limit_per_minute=ip_limit,
        trusted_proxy_count=trusted_proxy_count,
        credential_salt=salt,
    )
    return app


def _call(
    app: Starlette,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    async def _go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            return await c.request(method, path, headers=headers or {})

    return asyncio.run(_go())
