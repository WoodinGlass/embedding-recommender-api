"""Index promotion and rollback.

The index lifecycle state machine (``building`` -> ``active`` ->
``retired``) and the ``pg_advisory_lock`` that serialises promotions are
defined in ``docs/retrieval-and-evaluation.md`` § 11 and
``docs/ops.md`` § 5.2. This module holds the shared logic for promote and
rollback; the two CLI scripts are thin wrappers.

The advisory lock is held for the duration of the transaction and released
explicitly. If the process crashes, the connection closes and the lock is
released by the server; the transaction rolls back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from recsys.monitoring.logging import get_logger

log = get_logger(__name__)

#: Key for `pg_advisory_lock`. Fixed so two concurrent promotes in the
#: same database serialise. `hashtext` is a PostgreSQL built-in; the string
#: is chosen to be distinct from any other advisory lock the project may
#: take in the future.
ADVISORY_LOCK_KEY = "recsys.index_promote"


class PromoteError(Exception):
    """Raised when a promote or rollback cannot proceed."""


@dataclass(frozen=True)
class PromoteResult:
    index_version: str
    previous_index_version: str | None
    status: Literal["promoted", "already_active"]
    duration_ms: int


def _acquire_advisory_lock(connection: Any, *, timeout_s: float = 30.0) -> None:
    """Acquire the promote advisory lock, blocking up to ``timeout_s``.

    Uses the blocking form of ``pg_advisory_lock`` in a tight loop with a
    statement timeout, so that a stuck holder surfaces as a
    :class:`PromoteError` rather than hanging forever. The lock is
    session-scoped; it is released in :func:`_release_advisory_lock` or by
    the server when the connection closes.
    """
    deadline = time.monotonic() + timeout_s
    with connection.cursor() as cur:
        while True:
            cur.execute(
                "SELECT pg_try_advisory_lock(hashtext(%s))",
                (ADVISORY_LOCK_KEY,),
            )
            row = cur.fetchone()
            if row is not None and row[0]:
                return
            if time.monotonic() >= deadline:
                raise PromoteError(
                    f"could not acquire promote lock within {timeout_s:.1f}s; "
                    f"another promote or rollback may be in progress"
                )
            time.sleep(0.25)


def _release_advisory_lock(connection: Any) -> None:
    with connection.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(hashtext(%s))",
            (ADVISORY_LOCK_KEY,),
        )


def _release_advisory_lock_quietly(connection: Any) -> None:
    """Best-effort release of the advisory lock, logging on failure.

    Used from ``finally`` blocks where the surrounding code is already
    handling an exception (or where the lock is a convenience for a fast
    retry). A failure to release is not itself fatal: the server releases
    the lock when the session ends. Logging at debug records the case
    without masking whatever the outer block is doing. See ADR-0012 and
    ``docs/ops.md`` § 5.2.
    """
    try:
        _release_advisory_lock(connection)
    except Exception as e:
        log.debug(
            "index.promote.lock_release_failed",
            error_type=type(e).__name__,
            error_message=str(e),
        )


def _active_index_version(connection: Any) -> str | None:
    with connection.cursor() as cur:
        cur.execute("SELECT index_version FROM index_registry WHERE status = 'active'")
        row = cur.fetchone()
    return None if row is None else str(row[0])


def promote(
    connection: Any,
    *,
    target_index_version: str,
    timeout_s: float = 30.0,
) -> PromoteResult:
    """Make ``target_index_version`` the active index.

    Preconditions (checked inside the transaction, after the advisory lock
    is acquired):

    - The target exists in ``index_registry``.
    - The target's status is ``building`` (a brand-new index) or
      ``retired`` (re-promoting after a rollback). Promoting an ``active``
      index is a no-op and reported as ``already_active``.
    - The target's ``row_count`` equals the actual number of rows in
      ``embedding`` for that index. A mismatch indicates an incomplete
      build and is refused.

    Postconditions:

    - The previous active index is ``retired`` (``retired_at`` set).
    - The target is ``active`` (``activated_at`` set).
    - Exactly one row has ``status = 'active'`` at all times. The partial
      unique index enforces this; the advisory lock makes conflicts
      serialise rather than race.
    """
    t0 = time.perf_counter()
    _acquire_advisory_lock(connection, timeout_s=timeout_s)
    try:
        current = _active_index_version(connection)
        if current == target_index_version:
            log.info(
                "index.promote.already_active",
                index_version=target_index_version,
            )
            connection.commit()
            return PromoteResult(
                index_version=target_index_version,
                previous_index_version=None,
                status="already_active",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )

        with connection.cursor() as cur:
            cur.execute(
                "SELECT status, row_count FROM index_registry WHERE index_version = %s",
                (target_index_version,),
            )
            row = cur.fetchone()
        if row is None:
            raise PromoteError(f"index {target_index_version!r} not found in registry")
        status, registered_rows = str(row[0]), int(row[1])
        if status not in ("building", "retired"):
            raise PromoteError(
                f"index {target_index_version!r} has status {status!r}; "
                f"only 'building' or 'retired' can be promoted"
            )

        with connection.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM embedding WHERE index_version = %s",
                (target_index_version,),
            )
            count_row = cur.fetchone()
        if count_row is None:
            # SELECT count(*) always returns exactly one row; a None here
            # means the connection is in a state the driver did not expect.
            raise PromoteError(f"count query returned no row for {target_index_version!r}")
        actual_rows = int(count_row[0])
        if registered_rows != actual_rows:
            raise PromoteError(
                f"index {target_index_version!r} has row_count={registered_rows} "
                f"in the registry but {actual_rows} embeddings; the build is "
                f"incomplete and cannot be promoted"
            )

        # Retire the previous active index, if any.
        if current is not None:
            with connection.cursor() as cur:
                cur.execute(
                    "UPDATE index_registry "
                    "SET status = 'retired', retired_at = now() "
                    "WHERE index_version = %s",
                    (current,),
                )

        # Activate the target.
        with connection.cursor() as cur:
            cur.execute(
                "UPDATE index_registry "
                "SET status = 'active', activated_at = now(), "
                "    retired_at = NULL "
                "WHERE index_version = %s",
                (target_index_version,),
            )

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        _release_advisory_lock_quietly(connection)

    duration_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "index.promote.done",
        index_version=target_index_version,
        previous_index_version=current,
        duration_ms=duration_ms,
    )
    return PromoteResult(
        index_version=target_index_version,
        previous_index_version=current,
        status="promoted",
        duration_ms=duration_ms,
    )


def rollback(
    connection: Any,
    *,
    timeout_s: float = 30.0,
) -> PromoteResult:
    """Revert to the most recently retired index.

    Semantically identical to a promote of the most recent retired index.
    A convenience for operators; the ADR calls this a pointer swap.
    """
    _acquire_advisory_lock(connection, timeout_s=timeout_s)
    try:
        with connection.cursor() as cur:
            cur.execute(
                "SELECT index_version FROM index_registry "
                "WHERE status = 'retired' "
                "ORDER BY retired_at DESC NULLS LAST "
                "LIMIT 1"
            )
            row = cur.fetchone()
    finally:
        _release_advisory_lock_quietly(connection)

    if row is None:
        raise PromoteError("no retired index to roll back to")
    target = str(row[0])
    return promote(connection, target_index_version=target, timeout_s=timeout_s)
