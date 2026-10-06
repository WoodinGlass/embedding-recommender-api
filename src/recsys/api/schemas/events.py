"""Event ingestion schemas — see ``docs/contracts.md`` § 1.1."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExperimentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    variant: str = Field(min_length=1, max_length=64)


class EventEnvelope(BaseModel):
    """A single interaction event.

    Idempotent by ``event_id``. Server behavior (skew window, PII hashing) is
    specified in ``docs/contracts.md`` § 1.1 and enforced at the endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=8, max_length=64)
    event_ts: datetime
    event_type: Literal["impression", "click", "conversion"]
    user_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    request_id: str | None = Field(default=None, max_length=64)
    experiment: ExperimentRef | None = None
    position: int | None = Field(default=None, ge=1)
    value: float | None = None


class EventAck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: int = Field(ge=0)
    duplicates: int = Field(ge=0)
