"""Fallback chain — tiers 3 and 4 (ADR-0020).

The chain in ADR-0020 has four tiers plus 503:

1. ``cache``          — Redis cache hit (ADR-0015).
2. ``ann``            — ANN retrieval + re-rank.
3. ``fallback_ann``   — popular items read from the database.
4. ``fallback_cached``— the same list, held in process memory.
5. 503 ``none``       — every tier failed or produced an empty list.

This module implements tiers 3 and 4 as one pure function. Tier 1
(Redis) and tier 2 (ANN) belong to different layers: tier 2 lives in
``api.pipeline.sync_pipeline`` and tier 1 lives in the handler (it
needs the async cache store, which cannot run inside the worker
thread). The handler calls ``serve_fallback`` only after tier 2 has
failed.

The chain is a plain sequence, not a data structure: the order is
fixed by the ADR, and a table of callables would obscure that the
tiers have different shapes (one takes a connection, one does not).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from recsys.monitoring.logging import get_logger
from recsys.popularity import PopularityCache, read_snapshot
from recsys.popularity.snapshot import DEFAULT_TIMEOUT_SECONDS

log = get_logger(__name__)

#: The ``meta.source`` values this module can return. Both are named
#: in ``docs/contracts.md`` § 2.1 and in ADR-0020.
FallbackSource = Literal["fallback_ann", "fallback_cached"]


@dataclass(frozen=True)
class FallbackResult:
    """What one of the fallback tiers produced.

    ``items`` is a tuple of ``(item_id, score)`` pairs in the order
    the tier returned them (rank ascending for both tiers). The
    scores are placeholders in ``(0, 1]`` (ADR-0020); they are not
    comparable to an ANN score.
    """

    source: FallbackSource
    items: tuple[tuple[str, float], ...]


def serve_fallback(
    *,
    connection: Any | None,
    popularity_cache: PopularityCache,
    k: int,
    filters: Mapping[str, str] | None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> FallbackResult | None:
    """Try tier 3 then tier 4; return the first non-empty result.

    A ``None`` return means neither tier produced a list: the caller
    serves the 503 from tier 5. An empty list is not an answer
    (ADR-0020 § "the chain tries tiers in order and stops at the
    first that produces a non-empty list").

    ``connection`` may be ``None`` when the pool is unavailable
    (the ANN path already failed for the same reason). Tier 3 is
    skipped in that case; tier 4 runs from process memory and does
    not need the database.

    Exceptions from tier 3 are caught: the ADR's contract is that a
    database failure is a *fall through* to tier 4, not a request
    failure. The exception is logged with its type; the message is
    not (it can carry SQL text with schema names).
    """
    if connection is not None:
        try:
            items = read_snapshot(
                connection,
                k=k,
                filters=filters,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            log.warning(
                "fallback.tier3_error",
                error_type=type(exc).__name__,
                k=k,
            )
        else:
            if items:
                log.info("fallback.served", source="fallback_ann", n_items=len(items))
                return FallbackResult(source="fallback_ann", items=tuple(items))
            log.debug("fallback.tier3_empty", k=k)

    items = popularity_cache.get(k=k, filters=filters)
    if items:
        log.info("fallback.served", source="fallback_cached", n_items=len(items))
        return FallbackResult(source="fallback_cached", items=tuple(items))

    log.warning("fallback.exhausted", k=k, cache_size=popularity_cache.size)
    return None


__all__ = [
    "FallbackResult",
    "FallbackSource",
    "serve_fallback",
]
