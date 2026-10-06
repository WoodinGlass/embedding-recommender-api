"""Recommendation API schemas — see ``docs/contracts.md`` § 2.1."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendFilters(BaseModel):
    """Allowlisted metadata filters. Unknown keys are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")

    category: str | None = None
    brand: str | None = None
    language: str | None = None


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    seed_item_ids: list[str] = Field(default_factory=list, max_length=50)
    k: int = Field(default=10, ge=1, le=100)
    filters: RecommendFilters | None = None


class RecommendItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)


class ExperimentAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    variant: str


class RecommendMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["cache", "ann", "fallback"]
    model_version: str
    index_version: str
    experiment: ExperimentAssignment | None = None


class RecommendResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    items: list[RecommendItem]
    meta: RecommendMeta
