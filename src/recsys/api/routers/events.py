"""Event ingestion endpoint (ADR-0018).

The handler is a thin wrapper: it pulls the connection from the
pool, hands the batch to ``recsys.events.ingest_events``, and maps
the ingester's outcomes to HTTP statuses. The interesting logic
(skew window, idempotency, PII hashing) lives in the ingester;
this handler exists so the request shape and error envelope are
consistent with the rest of the API.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from typing import Any

import anyio
from fastapi import APIRouter, HTTPException, Request, status

from recsys.api.deps import DbPoolDep, PrincipalDep, RequestIdDep
from recsys.api.schemas.events import EventAck, EventBatch
from recsys.events.ingest import IngestValidationError, ingest_events
from recsys.monitoring.logging import get_logger

router = APIRouter(prefix="/v1", tags=["events"])

log = get_logger(__name__)

#: Skew window, from ADR-0018 § Skew window. An event older than
#: seven days or more than five minutes in the future aborts the
#: whole batch, not just the offending row.
_MAX_AGE_SECONDS = 604_800
_MAX_FUTURE_SECONDS = 300


def _sync_ingest(
    pool: object,
    *,
    events: list[Any],
    request_id: str,
    now: datetime,
    max_age_seconds: int,
    max_future_seconds: int,
    user_id_salt: str,
    user_id_salt_version: int,
) -> object:
    """Acquire a connection and run ``ingest_events`` (sync)."""
    with pool.connection() as conn:  # type: ignore[attr-defined]
        return ingest_events(
            conn,
            events=events,
            request_id=request_id,
            now=now,
            max_age_seconds=max_age_seconds,
            max_future_seconds=max_future_seconds,
            user_id_salt=user_id_salt,
            user_id_salt_version=user_id_salt_version,
        )


@router.post("/events", response_model=EventAck, status_code=202)
async def ingest_event(
    body: EventBatch,
    request: Request,
    principal: PrincipalDep,
    request_id: RequestIdDep,
    pool: DbPoolDep,
) -> EventAck:
    """Insert one batch of events; return the per-outcome counts.

    ``202 Accepted`` because the write may be a duplicate (the
    ``event_id`` is the idempotency key); the response describes
    what happened, and a retry is safe.
    """
    del principal  # auth only

    settings = request.app.state.settings
    limiter: anyio.CapacityLimiter = request.app.state.thread_limiter

    call = partial(
        _sync_ingest,
        pool,
        events=list(body.events),
        request_id=request_id,
        now=datetime.now(UTC),
        max_age_seconds=_MAX_AGE_SECONDS,
        max_future_seconds=_MAX_FUTURE_SECONDS,
        user_id_salt=str(getattr(settings, "user_id_hash_salt", "")),
        user_id_salt_version=int(getattr(settings, "user_id_hash_salt_version", 0)),
    )
    try:
        with anyio.fail_after(2.0):
            result = await anyio.to_thread.run_sync(call, limiter=limiter)
    except IngestValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "validation_error",
                "message": str(exc)[:200],
                "request_id": request_id,
            },
        ) from None
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "unavailable",
                "message": "event ingestion timeout",
                "request_id": request_id,
            },
        ) from None
    except Exception as exc:
        log.warning(
            "events.ingest_failed",
            request_id=request_id,
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "unavailable",
                "message": f"ingest failed: {type(exc).__name__}",
                "request_id": request_id,
            },
        ) from None

    log.info(
        "events.ingested",
        request_id=request_id,
        accepted=result.accepted,  # type: ignore[attr-defined]
        duplicates=result.duplicates,  # type: ignore[attr-defined]
        rejected=result.rejected,  # type: ignore[attr-defined]
    )
    return EventAck(
        accepted=result.accepted,  # type: ignore[attr-defined]
        duplicates=result.duplicates,  # type: ignore[attr-defined]
        rejected=result.rejected,  # type: ignore[attr-defined]
        generated_event_ids=result.generated_event_ids,  # type: ignore[attr-defined]
    )
