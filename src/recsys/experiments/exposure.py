"""Exposure writer (ADR-0017 § Exposure is logged, § Idempotent).

One row per served variant. The write is synchronous, on the same
PostgreSQL connection the rest of the serving path uses, into the
``experiment_exposure`` table migration 0002 created.

Why synchronous and not a queue: at the target traffic the insert
is a single statement on an indexed table, and a synchronous write
gives the caller an accurate answer about what was recorded. A
queue would make "accepted" mean "a worker might write it", which
is a weaker promise than the analysis can use (ADR-0017 § Exposure
is logged, and its Consequences).

Why the writer checks ``assignment.log_exposure``: a stopped or
disabled experiment must not record exposures (ADR-0017 § Fallback
variant, resolved to the status table — see
``recsys.experiments.assignment``). Doing the check in the writer
rather than at every call site means a future handler cannot forget
it. When the check fails the writer returns without touching the
database; it is a documented no-op, not an error.

Why the insert is wrapped in an explicit transaction: the same
reason ``PgRecencyProvider`` is (M3.4). Without it, psycopg begins
an implicit transaction on the insert, and a failure leaves the
connection in the aborted state for the next statement on the same
connection. The writer shares its connection with the retrieval
path, so a failed exposure must not poison the next query.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from recsys.events import UserIdHash
from recsys.experiments.assignment import Assignment

#: The event_id namespace. It keeps the exposure's id out of the
#: space a client-supplied event_id could collide with (ADR-0018 §
#: Idempotency: a client generates an opaque string; ``sha256`` of a
#: namespaced string cannot equal a UUID a client picked).
_EXPOSURE_ID_NAMESPACE = "exposure"


class ExposureWriteError(RuntimeError):
    """The exposure row could not be written.

    The caller (the M3.6 handler) catches this, increments
    ``recsys_experiment_exposure_errors_total{type}``, and returns
    the response anyway: an exposure write failure is a data-quality
    problem, not a request failure (ADR-0017 § Exposure is logged).
    """

    def __init__(self, *, cause: BaseException) -> None:
        super().__init__(f"exposure write failed: {type(cause).__name__}")
        self.cause = cause


@dataclass(frozen=True)
class ExposureResult:
    """The outcome of one write attempt.

    ``inserted`` is True when this call wrote a new row and False
    when it did not (either because the assignment's
    ``log_exposure`` was False, or because the id conflicted with an
    existing row from a retry). The two "False" cases are
    distinguishable by ``skipped_reason``.

    ``event_id`` is always computed, even when the write is
    skipped: a caller that wants to log the id for correlation has
    it without a second derivation.
    """

    event_id: str
    inserted: bool
    skipped_reason: str | None = None  # None | "log_exposure_false" | "duplicate"


def _exposure_event_id(*, experiment: str, request_id: str) -> str:
    """Return ``sha256(f"exposure:{experiment}:{request_id}")``.

    The formula is the one ``docs/contracts.md`` § 5 lists for the
    exposure write; a caller that derives its own id would produce a
    different uniqueness space and a retry would double-count.
    """
    if not experiment:
        raise ValueError("experiment must be a non-empty string")
    if not request_id:
        raise ValueError("request_id must be a non-empty string")
    payload = f"{_EXPOSURE_ID_NAMESPACE}:{experiment}:{request_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_exposure(
    *,
    connection: Any,
    assignment: Assignment,
    user_id_hash: UserIdHash,
    request_id: str,
    timeout_seconds: float = 0.5,
) -> ExposureResult:
    """Write one exposure row. Returns the outcome; never raises for
    a "not logged" case.

    Raises ``ExposureWriteError`` when the insert itself fails. The
    caller decides what to do with that (log, metric, response);
    the writer does not swallow it, because swallowing a write
    failure inside the writer would leave the caller unable to
    distinguish "not logged" from "tried and failed".

    ``connection`` is a psycopg connection (sync). ``timeout_seconds``
    is applied with ``set_config('statement_timeout', ..., true)``
    inside the transaction; the value is small because the write is a
    single indexed insert, and a slow one is a signal, not a workload
    to wait on. The function form (not ``SET LOCAL``) is required:
    PostgreSQL's extended-query protocol rejects a placeholder on the
    right-hand side of ``SET LOCAL ... = $1``.
    """
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")

    event_id = _exposure_event_id(experiment=assignment.experiment_name, request_id=request_id)

    if not assignment.log_exposure:
        # Stopped or disabled (ADR-0017 § Fallback variant resolved
        # to the status table). No database call.
        return ExposureResult(
            event_id=event_id,
            inserted=False,
            skipped_reason="log_exposure_false",
        )

    timeout_ms = max(1, int(timeout_seconds * 1000))
    try:
        with connection.transaction(), connection.cursor() as cur:
            cur.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                (str(timeout_ms),),
            )
            cur.execute(
                """
                    INSERT INTO experiment_exposure (
                        event_id, experiment, variant, request_id,
                        user_id_hash, user_id_hash_version, bucket,
                        effective_salt, paused, fallback_reason
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                (
                    event_id,
                    assignment.experiment_name,
                    assignment.variant,
                    request_id,
                    user_id_hash.hex,
                    user_id_hash.version,
                    assignment.bucket,
                    assignment.effective_salt,
                    assignment.paused,
                    assignment.fallback_reason,
                ),
            )
            inserted = cur.rowcount == 1
    except Exception as exc:
        raise ExposureWriteError(cause=exc) from exc

    return ExposureResult(
        event_id=event_id,
        inserted=inserted,
        skipped_reason=None if inserted else "duplicate",
    )


__all__ = [
    "ExposureResult",
    "ExposureWriteError",
    "write_exposure",
]
