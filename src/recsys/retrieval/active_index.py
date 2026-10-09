"""Active index version: fetch once at startup, refresh on a timer.

ADR-0015 (amended) explains why: ``PgvectorBackend.from_registry``
queries ``index_registry`` on every request, and the pgvector
extension version is stable for the process lifetime. Both values
move off the request path.

The refresh loop never resets the value to ``None``. A refresh that
fails (a database hiccup) keeps the previous version: a stale index
version is a bounded staleness, an absent one turns every request
into a fallback. The staleness gauge exposes the age.
"""

from __future__ import annotations

from typing import Any

import anyio

from recsys.monitoring.logging import get_logger

log = get_logger(__name__)

#: Seconds between refreshes. Bounds the staleness window after a
#: promote; an operator-facing value, not a tuning knob.
ACTIVE_INDEX_REFRESH_SECONDS: float = 30.0


def fetch_active_index_version(connection: Any) -> str | None:
    """Return the active row's ``index_version`` or ``None``.

    One query; the caller owns the connection's lifetime.
    """
    with connection.cursor() as cur:
        cur.execute("SELECT index_version FROM index_registry WHERE status = 'active'")
        row = cur.fetchone()
    return str(row[0]) if row is not None else None


def fetch_pgvector_version(connection: Any) -> str | None:
    """Return the installed pgvector extension version or ``None``.

    Read once at startup; the value does not change without an
    ``ALTER EXTENSION`` (which requires a restart to matter).
    """
    with connection.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    return str(row[0]) if row is not None else None


async def refresh_active_index(
    *,
    pool: Any,
    state: Any,
    interval_seconds: float = ACTIVE_INDEX_REFRESH_SECONDS,
) -> None:
    """Refresh ``state.active_index`` every ``interval_seconds``.

    Runs in the app lifespan. Never raises: a refresh failure is
    logged and the previous value is kept. Cancellation (shutdown)
    propagates via ``anyio.sleep``.
    """
    from recsys.monitoring.metrics import (
        ACTIVE_INDEX_REFRESH_FAILURES_TOTAL,
        ACTIVE_INDEX_STALENESS_SECONDS,
    )

    while True:
        try:
            await anyio.sleep(interval_seconds)
        except anyio.get_cancelled_exc_class():
            return

        try:
            with pool.connection() as conn:
                current = fetch_active_index_version(conn)
        except Exception as exc:
            ACTIVE_INDEX_REFRESH_FAILURES_TOTAL.inc()
            log.warning("active_index.refresh_failed", error=type(exc).__name__)
            continue

        ACTIVE_INDEX_STALENESS_SECONDS.set(0.0)

        if current is None:
            # No active row: keep the previous value. An empty
            # registry is a state an operator fixes, not one the
            # request path should react to by dropping to None.
            log.warning("active_index.refresh_empty")
            continue

        previous = getattr(state, "active_index", None)
        if current != previous:
            state.active_index = current
            log.info("active_index.changed", from_=previous, to=current)
        else:
            state.active_index = current


__all__ = [
    "ACTIVE_INDEX_REFRESH_SECONDS",
    "fetch_active_index_version",
    "fetch_pgvector_version",
    "refresh_active_index",
]
