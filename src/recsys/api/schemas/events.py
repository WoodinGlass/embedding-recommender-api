"""Event ingestion schemas — see ``docs/contracts.md`` § 1.1."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExperimentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    variant: str = Field(min_length=1, max_length=64)


class EventEnvelope(BaseModel):
    """A single interaction event.

    Idempotent by ``event_id``. Server behavior (skew window, PII
    hashing) is specified in ``docs/contracts.md`` § 1.1 and enforced
    at the endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=64,
        description=(
            "Client-generated unique id. If omitted, the server "
            "generates sha256(f'{request_id}:{index}') and returns it "
            "in the ack's `generated_event_ids`."
        ),
    )
    event_ts: datetime
    event_type: Literal["impression", "click", "conversion"]
    user_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    request_id: str | None = Field(default=None, max_length=64)
    experiment: ExperimentRef | None = None
    position: int | None = Field(default=None, ge=1)
    value: float | None = None


class EventBatch(BaseModel):
    """The ``POST /v1/events`` request body. One to one hundred events."""

    model_config = ConfigDict(extra="forbid")

    events: Annotated[list[EventEnvelope], Field(min_length=1, max_length=100)]


class EventAck(BaseModel):
    """The ``202 Accepted`` response body."""

    model_config = ConfigDict(extra="forbid")

    accepted: int = Field(ge=0)
    duplicates: int = Field(ge=0)
    rejected: int = Field(ge=0)
    generated_event_ids: list[str] | None = Field(
        default=None,
        description=(
            "Parallel to the subset of the request's events whose "
            "event_id the client omitted. null when the client "
            "supplied every id."
        ),
    )
