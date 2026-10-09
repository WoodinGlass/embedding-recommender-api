"""Batch event ingestion (ADR-0018 § Decision).

The function that turns a validated ``EventBatch`` into rows:

1. Validate the skew window per event. An event outside
   ``[now - max_age, now + max_future]`` aborts the whole batch with
   ``IngestValidationError``: the ADR says "rejected with 422 and an
   error detail naming the field and the offending timestamp", and
   the handler turns that into the response. A partial batch (99 of
   100 written, 1 rejected) is not what the ADR describes; the
   client fixes the one event and retries, and the retry is
   idempotent by `event_id`.
2. Hash the user id (HMAC-SHA256, versioned salt; ADR-0018).
3. Generate `event_id` for events the client did not supply one.
   The generated id is `sha256(f"{request_id}:{index}")`, stable for
   a given request and index, so a retry of the same request with
   the same `request_id` produces the same ids and the insert is
   idempotent.
4. Batch insert with `ON CONFLICT (event_id) DO NOTHING`. The
   difference between the row count and the batch size is the
   duplicate count; the ack reports it.

The function is pure with respect to the process: it opens no
connection, reads no config, writes no log, and consults no clock
(the caller passes `now`). That is what makes it testable without a
database and a frozen clock.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final

from recsys.api.schemas.events import EventEnvelope
from recsys.events.hashing import hash_user_id

#: The per-batch statement timeout. The insert is a single
#: `executemany` on an indexed table; a slow one is a signal, not a
#: workload to wait on.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 0.5


class IngestValidationError(ValueError):
    """One event in the batch failed validation.

    The whole batch is rejected. ``field`` names the offending field
    (``"event_ts"`` today; a future validation adds its own);
    ``index`` is the position in the batch, for the error detail.
    """

    def __init__(self, *, field: str, index: int, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.index = index


@dataclass(frozen=True)
class IngestResult:
    """The outcome of one batch.

    ``accepted`` is the number of rows actually inserted;
    ``duplicates`` the number the database already held (a retry);
    ``rejected`` is 0 today (a validation failure raises instead of
    producing a partial batch); ``generated_event_ids`` is the list
    of ids the server generated, parallel to the subset of the
    request's events whose `event_id` was `None`, or `None` when
    every event carried one.
    """

    accepted: int
    duplicates: int
    rejected: int
    generated_event_ids: list[str] | None


def _generate_event_id(*, request_id: str, index: int) -> str:
    """Return ``sha256(f"{request_id}:{index}")``.

    The formula is the one ADR-0018 § Idempotency names; a caller
    that derived its own id would produce a different uniqueness
    space and a retry would double-count. The output is 64 hex
    characters, which fits the schema's max length exactly.
    """
    return hashlib.sha256(f"{request_id}:{index}".encode()).hexdigest()


def _validate_skew(
    event_ts: datetime,
    *,
    index: int,
    now: datetime,
    max_age: timedelta,
    max_future: timedelta,
) -> None:
    if event_ts.tzinfo is None:
        raise IngestValidationError(
            field="event_ts",
            index=index,
            message=(
                "event_ts must be timezone-aware (ISO 8601 with an "
                "offset, e.g. '2026-01-01T00:00:00Z')"
            ),
        )
    delta = event_ts - now
    if delta < -max_age:
        raise IngestValidationError(
            field="event_ts",
            index=index,
            message=(
                f"event_ts is more than {max_age} before now; "
                "the skew window rejects an event this old"
            ),
        )
    if delta > max_future:
        raise IngestValidationError(
            field="event_ts",
            index=index,
            message=(
                f"event_ts is more than {max_future} after now; "
                "a client clock this far off is a misconfiguration"
            ),
        )


def ingest_events(
    connection: Any,
    *,
    events: list[EventEnvelope],
    request_id: str,
    now: datetime,
    max_age_seconds: int,
    max_future_seconds: int,
    user_id_salt: str,
    user_id_salt_version: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> IngestResult:
    """Insert the batch; return the counts and generated ids.

    Raises ``IngestValidationError`` when any event fails the skew
    window check; the caller (the handler) turns that into a 422
    with the offending field and index in the error detail. Raises
    the underlying exception for a database failure; the handler
    turns that into a 500 or a 503, per the handler's policy.
    """
    if not events:
        raise ValueError("events must be a non-empty list")
    if not request_id:
        raise ValueError("request_id must be a non-empty string")
    if max_age_seconds < 0:
        raise ValueError(f"max_age_seconds must be >= 0, got {max_age_seconds}")
    if max_future_seconds < 0:
        raise ValueError(f"max_future_seconds must be >= 0, got {max_future_seconds}")
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")

    max_age = timedelta(seconds=max_age_seconds)
    max_future = timedelta(seconds=max_future_seconds)

    rows: list[tuple[Any, ...]] = []
    generated_ids: list[str] = []
    for i, env in enumerate(events):
        _validate_skew(env.event_ts, index=i, now=now, max_age=max_age, max_future=max_future)
        h = hash_user_id(env.user_id, salt=user_id_salt, salt_version=user_id_salt_version)
        if env.event_id is None:
            eid = _generate_event_id(request_id=request_id, index=i)
            generated_ids.append(eid)
        else:
            eid = env.event_id
        rows.append(
            (
                eid,
                env.event_ts,
                env.event_type,
                h.hex,
                h.version,
                env.item_id,
                env.request_id,
                env.experiment.name if env.experiment is not None else None,
                env.experiment.variant if env.experiment is not None else None,
                env.position,
                env.value,
            )
        )

    timeout_ms = max(1, int(timeout_seconds * 1000))
    with connection.transaction(), connection.cursor() as cur:
        # The function form (not ``SET LOCAL ... = $1``) is
        # required: PostgreSQL's extended-query protocol rejects
        # a placeholder on the right-hand side of ``SET LOCAL``.
        # See ``PgRecencyProvider`` for the same fix in M3.4.
        cur.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(timeout_ms),),
        )
        cur.executemany(
            """
                INSERT INTO event (
                    event_id, event_ts, event_type,
                    user_id_hash, user_id_hash_version, item_id,
                    request_id, experiment_name, experiment_variant,
                    position, value
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
            rows,
        )
        # psycopg sets ``rowcount`` to the total rows affected
        # across the batch. With ``ON CONFLICT DO NOTHING`` that
        # is the number of new rows, not the batch size; the
        # difference is the duplicates the database already
        # held.
        inserted = cur.rowcount if cur.rowcount is not None else 0

    duplicates = len(rows) - inserted
    return IngestResult(
        accepted=inserted,
        duplicates=duplicates,
        rejected=0,
        generated_event_ids=generated_ids or None,
    )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "IngestResult",
    "IngestValidationError",
    "ingest_events",
]
