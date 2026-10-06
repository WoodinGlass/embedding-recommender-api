"""Churn scoring schemas — see ``docs/contracts.md`` § 2.4."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChurnScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)


class ChurnScoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    user_id: str
    score: float = Field(ge=0.0, le=1.0)
    model_version: str
    features_ts: datetime
