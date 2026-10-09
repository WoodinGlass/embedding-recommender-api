"""Tier 4 in-memory popularity cache (ADR-0020 § Tier 4).

The cache is a snapshot of the `popularity_snapshot` table loaded
from a JSON file on disk. The file is written by
`scripts/refresh_popularity.py` alongside the table refresh, so the
cache has a source that does not depend on the database being up at
process start: a process that boots while PostgreSQL is down reads
the last known good file and can still serve tier 4.

The cache is immutable. There is no background refresh task; the
operator runs `make popularity-refresh` (which rewrites both the
table and the file) and restarts the process (or the process is
restarted by the deploy that follows the refresh). This is the same
trade-off ADR-0022 makes for the hot config: a change requires a
restart, which is simpler than a mid-process flip and has a
narrower set of failure modes.

Filtering is a comprehension over the in-memory list. At N = 1000 a
filter is microseconds; even at the 100 000 upper bound the
comprehension is bounded by the size the operator chose.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from recsys.retrieval.filters import FILTER_FIELDS

#: Only this schema version of the file is accepted. A version bump
#: is a code change; the file is not silently upgraded.
SUPPORTED_SCHEMA_VERSION: Final[int] = 1


class PopularityCacheError(RuntimeError):
    """The file is missing, malformed, or has an unsupported version."""


@dataclass(frozen=True)
class SnapshotItem:
    """One entry of the popularity snapshot, as written by refresh."""

    item_id: str
    rank: int
    category: str
    brand: str
    language: str


class PopularityCache:
    """A read-only, in-memory copy of the popularity snapshot.

    Instances are immutable; `empty()` and `from_file()` are the
    only constructors. `get` returns at most `k` `(item_id, score)`
    pairs; the score formula matches the tier-3 reader so a client
    sees the same scale regardless of which fallback tier served
    the request.
    """

    def __init__(self, items: tuple[SnapshotItem, ...]) -> None:
        self._items = items
        # The max rank is the denominator of the score formula; the
        # file is authoritative about what ranks exist, so the cache
        # does not re-scan on every request.
        self._max_rank = max((it.rank for it in items), default=0)

    @classmethod
    def empty(cls) -> PopularityCache:
        """An empty cache. Tier 4 falls through to tier 5."""
        return cls(())

    @classmethod
    def from_file(cls, path: pathlib.Path) -> PopularityCache:
        """Read a cache from `path`. Raises PopularityCacheError."""
        if not path.is_file():
            raise PopularityCacheError(f"popularity cache file not found: {path}")
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PopularityCacheError(f"popularity cache file is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise PopularityCacheError("popularity cache root must be an object")
        if doc.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
            raise PopularityCacheError(
                f"popularity cache schema_version must be {SUPPORTED_SCHEMA_VERSION}, "
                f"got {doc.get('schema_version')!r}"
            )
        raw_items = doc.get("items")
        if not isinstance(raw_items, list):
            raise PopularityCacheError("popularity cache 'items' must be a list")
        items: list[SnapshotItem] = []
        for i, entry in enumerate(raw_items):
            if not isinstance(entry, dict):
                raise PopularityCacheError(f"items[{i}] must be an object")
            try:
                items.append(
                    SnapshotItem(
                        item_id=str(entry["item_id"]),
                        rank=int(entry["rank"]),
                        category=str(entry["category"]),
                        brand=str(entry["brand"]),
                        language=str(entry["language"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PopularityCacheError(f"items[{i}] is malformed: {exc}") from exc
        return cls(tuple(items))

    @property
    def size(self) -> int:
        return len(self._items)

    def get(
        self,
        *,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(item_id, score)`` pairs.

        The snapshot is ordered by rank ascending; ties cannot occur
        because rank is the primary ordering of the file and the
        refresh writes a strict order. The filters are applied as an
        equality check per `FILTER_FIELDS`; a field absent from the
        filter is not constrained.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}")

        active_filters = {
            f: (filters or {}).get(f) for f in FILTER_FIELDS if (filters or {}).get(f) is not None
        }

        result: list[tuple[str, float]] = []
        for item in self._items:
            if active_filters:
                ok = True
                for field, want in active_filters.items():
                    if getattr(item, field) != want:
                        ok = False
                        break
                if not ok:
                    continue
            score = 1.0 - item.rank / (self._max_rank + 1)
            result.append((item.item_id, score))
            if len(result) >= k:
                break
        return result


__all__ = [
    "SUPPORTED_SCHEMA_VERSION",
    "PopularityCache",
    "PopularityCacheError",
    "SnapshotItem",
]
