"""Event ingestion endpoint.

Stub returns ``503``; the ingestion handler lands in M3.6.6e and the
A/B analysis that reads the events in M5.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from recsys.api.deps import PrincipalDep
from recsys.api.schemas.events import EventAck, EventEnvelope

router = APIRouter(prefix="/v1", tags=["events"])


@router.post("/events", response_model=EventAck, status_code=202)
async def ingest_event(_event: EventEnvelope, principal: PrincipalDep) -> EventAck:
    del principal  # auth only; the handler lands in M3.6.6e
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Event ingestion lands in M3.",
    )
