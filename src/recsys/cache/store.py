"""Cache-aside store for retrieval results (ADR-0015).

The store sits between the handler and the retrieval backend. On a
hit, the handler skips retrieval; on a miss, the handler runs
retrieval and writes the result. The store never caches the
re-ranked list: the re-ranker's recency term is a function of
wall-clock time, so a cached ordering would be stale by construction.
Only the retrieval output (the candidate list) is stored.

Failure behavior, from ADR-0015:

- A cache read that raises is logged, counted as ``error``, and
  treated as a miss. The request proceeds to retrieval. A cache is a
  performance optimization, not a correctness mechanism; a failed
  read must not become a failed request.
- When the shared Redis breaker is open, the store bypasses Redis
  entirely and reports ``bypass``. It does not retry, and it does not
  pay the timeout — that is the point of the breaker.
- A negative lookup (an item that does not exist) is stored as a
  sentinel for the negative TTL. A caller who probes a nonexistent
  id repeatedly does not hit the retrieval path on every request.

The store does **not** import ``monitoring.metrics`` at module load:
the counter names are looked up lazily inside the methods that emit
them. That keeps the import graph acyclic in either direction and
lets a future refactor move the metrics without touching the store's
public surface.

The Redis client is created on first use (lazy), with the timeout
passed in the constructor. ``socket_timeout`` is where the timeout
lives, not ``asyncio.wait_for``: cancellation from ``wait_for``
leaves a redis-py connection in a state the pool does not always
recover cleanly, while ``socket_timeout`` is handled by redis-py
itself. See ADR-0015 (D5).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from recsys.cache.ttl import NEGATIVE_SENTINEL
from recsys.monitoring.logging import get_logger
from recsys.resilience.breaker import CircuitBreaker, CircuitOpenError

log = get_logger(__name__)


class CacheLookup(StrEnum):
    """What the store found for a key.

    The five outcomes map to four Prometheus results plus one
    dedicated counter:

    - ``HIT`` -> ``result="hit"``
    - ``NEGATIVE`` -> ``result="hit"`` **and**
      ``recsys_cache_negative_hits_total``
    - ``MISS`` -> ``result="miss"``
    - ``BYPASS`` -> ``result="bypass"``
    - ``ERROR`` -> ``result="error"``
    """

    HIT = "hit"
    NEGATIVE = "negative"
    MISS = "miss"
    BYPASS = "bypass"
    ERROR = "error"


@dataclass(frozen=True)
class CacheResult:
    """The store's answer for one lookup.

    ``items`` is a tuple of ``(item_id, score)`` pairs for ``HIT``
    and ``None`` for every other outcome. The tuple (not list) is
    deliberate: the store must not hand back a value a caller can
    mutate and poison the cache with.
    """

    outcome: CacheLookup
    items: tuple[tuple[str, float], ...] | None = None


def _serialize(items: Sequence[tuple[str, float]]) -> bytes:
    """Return a compact, deterministic byte payload for ``items``."""
    return json.dumps(
        [[item_id, float(score)] for item_id, score in items],
        separators=(",", ":"),
    ).encode("utf-8")


def _deserialize(raw: bytes) -> tuple[tuple[str, float], ...]:
    """Parse a cached payload back into ``(item_id, score)`` tuples.

    Raises ``ValueError`` (or ``TypeError`` / ``KeyError`` from the
    comprehension) if the payload is not a list of pairs. The caller
    treats that as an ``ERROR`` outcome: a corrupted entry is not a
    hit and not a miss.
    """
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("cache payload is not a list")
    out: list[tuple[str, float]] = []
    for entry in parsed:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError("cache payload entry is not a [id, score] pair")
        item_id, score = entry
        if not isinstance(item_id, str):
            raise ValueError("cache payload item id is not a string")
        out.append((item_id, float(score)))
    return tuple(out)


class CacheStore:
    """Cache-aside store backed by Redis, guarded by a shared breaker.

    The breaker is passed in, not built here: it is shared with the
    rate limiter because Redis is one dependency (ADR-0015).
    """

    def __init__(
        self,
        *,
        redis_url: str,
        breaker: CircuitBreaker,
        socket_timeout_seconds: float = 0.1,
    ) -> None:
        if socket_timeout_seconds <= 0:
            raise ValueError(f"socket_timeout_seconds must be > 0, got {socket_timeout_seconds}")
        self._redis_url = redis_url
        self._breaker = breaker
        self._socket_timeout = socket_timeout_seconds

        self._redis: Any = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def close(self) -> None:
        """Close the Redis connection. Safe to call more than once."""
        if self._redis is not None:
            with contextlib.suppress(Exception):
                await self._redis.aclose()
            self._redis = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    async def lookup(self, key: str) -> CacheResult:
        """Return the cached result for ``key``.

        Never raises for a dependency problem: an open breaker, a
        connection error, and a corrupted entry all produce a
        non-hit outcome the caller treats as "no cached value".
        """
        if self._breaker.is_open():
            log.debug("cache.bypass", reason="breaker_open", key=key)
            self._emit(CacheLookup.BYPASS)
            return CacheResult(CacheLookup.BYPASS)

        try:
            raw = await self._breaker.call(self._redis_get(key))
        except CircuitOpenError:
            # The breaker flipped between is_open() and call(), or a
            # probe is in flight. Either way: bypass, do not pay the
            # timeout.
            log.debug("cache.bypass", reason="breaker_open", key=key)
            self._emit(CacheLookup.BYPASS)
            return CacheResult(CacheLookup.BYPASS)
        except Exception as exc:
            log.warning("cache.error", error_type=type(exc).__name__, key=key)
            self._emit(CacheLookup.ERROR)
            return CacheResult(CacheLookup.ERROR)

        if raw is None:
            log.debug("cache.miss", key=key)
            self._emit(CacheLookup.MISS)
            return CacheResult(CacheLookup.MISS)

        if raw == NEGATIVE_SENTINEL:
            log.debug("cache.hit", key=key, negative=True)
            self._emit(CacheLookup.NEGATIVE)
            return CacheResult(CacheLookup.NEGATIVE)

        try:
            items = _deserialize(raw)
        except (ValueError, TypeError, KeyError) as exc:
            # A corrupted entry: log, count as error, treat as miss.
            log.warning("cache.error", error_type=type(exc).__name__, key=key)
            self._emit(CacheLookup.ERROR)
            return CacheResult(CacheLookup.ERROR)

        log.debug("cache.hit", key=key, item_count=len(items))
        self._emit(CacheLookup.HIT)
        return CacheResult(CacheLookup.HIT, items=items)

    async def store(
        self,
        key: str,
        items: Sequence[tuple[str, float]],
        *,
        ttl_seconds: int,
    ) -> bool:
        """Write ``items`` under ``key``. Returns True on success.

        A write failure is logged and counted; it is not raised. The
        caller's response is already correct (it has the retrieval
        result); a failed write only means the next caller pays for
        the same work.
        """
        payload = _serialize(items)
        return await self._write(key, payload, ttl_seconds=ttl_seconds)

    async def store_negative(self, key: str, *, ttl_seconds: int) -> bool:
        """Write the negative sentinel under ``key``. Returns True on success."""
        return await self._write(key, NEGATIVE_SENTINEL, ttl_seconds=ttl_seconds)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    async def _write(self, key: str, payload: bytes, *, ttl_seconds: int) -> bool:
        if self._breaker.is_open():
            return False
        try:
            return await self._breaker.call(self._redis_set(key, payload, ttl_seconds))
        except CircuitOpenError:
            return False
        except Exception as exc:
            from recsys.monitoring.metrics import CACHE_WRITE_ERRORS_TOTAL

            log.warning("cache.error", error_type=type(exc).__name__, key=key, op="set")
            CACHE_WRITE_ERRORS_TOTAL.labels(type=type(exc).__name__).inc()
            return False

    async def _redis_get(self, key: str) -> bytes | None:
        await self._ensure_redis()
        # `self._redis` is Any (the client class is loaded lazily, and
        # importing redis stubs here would force every caller to
        # install the extra). The annotation documents the contract.
        raw: bytes | None = await self._redis.get(key)
        return raw

    async def _redis_set(self, key: str, payload: bytes, ttl_seconds: int) -> bool:
        await self._ensure_redis()
        if ttl_seconds < 1:
            return False
        await self._redis.set(key, payload, ex=ttl_seconds)
        return True

    async def _ensure_redis(self) -> None:
        if self._redis is not None:
            return
        import redis.asyncio as aioredis

        # See ``limiter.py`` for why this call carries a mypy
        # suppression: redis-py 5.x declares ``from_url`` with
        # ``**kwargs: Any``, which mypy strict reports as untyped.
        self._redis = aioredis.from_url(  # type: ignore[no-untyped-call]
            self._redis_url,
            decode_responses=False,
            socket_timeout=self._socket_timeout,
            socket_connect_timeout=self._socket_timeout,
        )

    def _emit(self, outcome: CacheLookup) -> None:
        """Emit the cache metric for ``outcome``.

        The metric import is lazy so the module does not depend on
        the monitoring layer at import time. ``NEGATIVE`` counts as
        a hit **and** bumps the negative-hits counter, matching the
        ADR-0015 wording ("a sentinel hit is a hit with a null
        payload").
        """
        from recsys.monitoring.metrics import CACHE_NEGATIVE_HITS, CACHE_REQUESTS

        if outcome is CacheLookup.NEGATIVE:
            CACHE_REQUESTS.labels(result="hit").inc()
            CACHE_NEGATIVE_HITS.inc()
            return
        CACHE_REQUESTS.labels(result=outcome.value).inc()
