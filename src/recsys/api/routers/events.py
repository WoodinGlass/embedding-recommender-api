"""Event ingestion endpoint.

Stub returns ``503``; implementation lands in M3 (endpoint) and M5 (analysis).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from recsys.api.schemas.events import EventAck, EventEnvelope

router = APIRouter(prefix="/v1", tags=["events"])


@router.post("/events", response_model=EventAck, status_code=202)
async def ingest_event(_event: EventEnvelope) -> EventAck:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Event ingestion lands in M3.",
    )
