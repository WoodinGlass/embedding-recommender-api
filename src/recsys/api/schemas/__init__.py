"""Pydantic schemas mirroring ``docs/contracts.md``."""

from recsys.api.schemas.churn import ChurnScoreRequest, ChurnScoreResponse
from recsys.api.schemas.common import ErrorBody, ErrorCode, ErrorEnvelope
from recsys.api.schemas.events import EventAck, EventEnvelope, ExperimentRef
from recsys.api.schemas.recommend import (
    ExperimentAssignment,
    RecommendFilters,
    RecommendItem,
    RecommendMeta,
    RecommendRequest,
    RecommendResponse,
)

__all__ = [
    "ChurnScoreRequest",
    "ChurnScoreResponse",
    "ErrorBody",
    "ErrorCode",
    "ErrorEnvelope",
    "EventAck",
    "EventEnvelope",
    "ExperimentAssignment",
    "ExperimentRef",
    "RecommendFilters",
    "RecommendItem",
    "RecommendMeta",
    "RecommendRequest",
    "RecommendResponse",
]
