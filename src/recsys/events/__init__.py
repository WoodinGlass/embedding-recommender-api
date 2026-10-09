"""Event ingestion primitives (ADR-0018)."""

from recsys.events.hashing import UserIdHash, hash_user_id

__all__ = [
    "UserIdHash",
    "hash_user_id",
]
