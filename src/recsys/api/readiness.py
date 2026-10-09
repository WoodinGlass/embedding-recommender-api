"""Readiness checks (ADR-0019).

Four checks, one aggregate. ``db`` and ``index`` are required;
``redis`` is optional; ``encoder`` is required in prod only. The
aggregate is ``ready`` (all pass), ``degraded`` (required pass,
optional fail), or ``not_ready`` (a required check fails).

Each ``check_*`` returns a dict with at least ``ok: bool``. A
failed check carries an ``error`` from a closed vocabulary:
``timeout``, ``connection_refused``, ``auth_failed``,
``not_configured``, ``unknown``. The vocabulary is the one
``docs/contracts.md`` section 2.6 fixes.
"""

from __future__ import annotations

import time
from typing import Any

# --- error vocabulary ------------------------------------------------- #
TIMEOUT = "timeout"
CONNECTION_REFUSED = "connection_refused"
AUTH_FAILED = "auth_failed"
NOT_CONFIGURED = "not_configured"
UNKNOWN = "unknown"


def classify_error(exc: BaseException) -> str:
    """Map an exception to the closed error vocabulary."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg:
        return TIMEOUT
    if "refused" in msg:
        return CONNECTION_REFUSED
    if "auth" in msg or "password" in msg:
        return AUTH_FAILED
    if "connection" in name or "operational" in name:
        return CONNECTION_REFUSED
    return UNKNOWN


def check_db(pool: Any, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
    """``SELECT 1`` on a pooled connection."""
    start = time.monotonic()
    try:
        with pool.connection(timeout=timeout_seconds) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception as exc:
        return {"ok": False, "required": True, "error": classify_error(exc)}
    return {
        "ok": True,
        "required": True,
        "latency_ms": int((time.monotonic() - start) * 1000),
    }


def check_index(pool: Any, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
    """Active index row + at least one embedding row for it."""
    start = time.monotonic()
    try:
        with pool.connection(timeout=timeout_seconds) as conn, conn.cursor() as cur:
            cur.execute("SELECT index_version FROM index_registry WHERE status = 'active'")
            row = cur.fetchone()
            if row is None:
                return {
                    "ok": False,
                    "required": True,
                    "error": UNKNOWN,
                    "index_version": None,
                }
            index_version = str(row[0])
            cur.execute(
                "SELECT COUNT(*) FROM embedding WHERE index_version = %s",
                (index_version,),
            )
            row = cur.fetchone()
            count = int(row[0]) if row is not None else 0
    except Exception as exc:
        return {"ok": False, "required": True, "error": classify_error(exc)}

    latency_ms = int((time.monotonic() - start) * 1000)
    if count == 0:
        return {
            "ok": False,
            "required": True,
            "error": UNKNOWN,
            "index_version": index_version,
            "row_count": 0,
            "latency_ms": latency_ms,
        }
    return {
        "ok": True,
        "required": True,
        "index_version": index_version,
        "row_count": count,
        "latency_ms": latency_ms,
    }


async def check_redis(url: str, *, timeout_seconds: float = 0.5) -> dict[str, Any]:
    """``PING`` on a short-lived Redis client."""
    start = time.monotonic()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(  # type: ignore[no-untyped-call]
            url,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
        )
        try:
            pong = await client.ping()
        finally:
            await client.aclose()
    except Exception as exc:
        return {"ok": False, "required": False, "error": classify_error(exc)}
    if not pong:
        return {"ok": False, "required": False, "error": UNKNOWN}
    return {
        "ok": True,
        "required": False,
        "latency_ms": int((time.monotonic() - start) * 1000),
    }


def aggregate(checks: dict[str, dict[str, Any]]) -> tuple[str, int]:
    """Reduce the checks to ``(status, http_code)``.

    ``ready`` when every check passes; ``degraded`` when a required
    check passes but an optional one does not; ``not_ready`` when a
    required check fails. HTTP is 200 for ready and degraded, 503
    for not_ready — a Redis hiccup must not drain a healthy
    instance (ADR-0019).
    """
    required = [c for c in checks.values() if c.get("required") is True]
    optional = [c for c in checks.values() if c.get("required") is not True]

    if not all(c.get("ok") is True for c in required):
        return "not_ready", 503
    if all(c.get("ok") is True for c in optional):
        return "ready", 200
    return "degraded", 200


__all__ = [
    "aggregate",
    "check_db",
    "check_index",
    "check_redis",
    "classify_error",
]
