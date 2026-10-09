"""Event ingestion primitives (ADR-0018)."""

from recsys.events.hashing import UserIdHash, hash_user_id
from recsys.events.ingest import (
    DEFAULT_TIMEOUT_SECONDS,
    IngestResult,
    IngestValidationError,
    ingest_events,
)

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "IngestResult",
    "IngestValidationError",
    "UserIdHash",
    "hash_user_id",
    "ingest_events",
]
