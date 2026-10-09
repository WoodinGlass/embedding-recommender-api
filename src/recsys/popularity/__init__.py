"""Popularity snapshot for the fallback chain (ADR-0020)."""

from recsys.popularity.cache import (
    SUPPORTED_SCHEMA_VERSION,
    PopularityCache,
    PopularityCacheError,
    SnapshotItem,
)
from recsys.popularity.refresh import (
    DEFAULT_REFRESH_TIMEOUT_SECONDS,
    fetch_snapshot_rows,
    refresh_popularity_snapshot,
    write_popularity_file,
)
from recsys.popularity.snapshot import (
    read_snapshot,
)

__all__ = [
    "DEFAULT_REFRESH_TIMEOUT_SECONDS",
    "SUPPORTED_SCHEMA_VERSION",
    "PopularityCache",
    "PopularityCacheError",
    "SnapshotItem",
    "fetch_snapshot_rows",
    "read_snapshot",
    "refresh_popularity_snapshot",
    "write_popularity_file",
]
