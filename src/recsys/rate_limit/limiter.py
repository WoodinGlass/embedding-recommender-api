"""Redis token bucket with a per-instance fallback (ADR-0014).

The design:

- One key per ``(bucket_id, class)``. The key lives in Redis as a hash
  with two fields (``tokens`` and ``last``), refilled by the Lua script
  on every check. Memory is O(1) per key regardless of request rate.
- One round trip per check: the script does the read-modify-write
  server-side, so there is no race between the read and the write and
  no separate reap step.
- A per-instance in-memory bucket when Redis is unreachable. The
  fallback's limit is the configured limit divided by
  ``INSTANCE_COUNT`` so the aggregate across instances stays close to
  the configured number.
- **A shared circuit breaker (ADR-0015) is consulted before every
  Redis call.** The breaker lives under ``resilience/`` and is shared
  with the response cache because Redis is one dependency. When the
  breaker is open the limiter goes straight to its per-instance
  fallback; a Redis that is down is not probed per request.

The Lua script is a string constant, not a file on disk: one file to
review for how the limiter works, no package-data configuration to
keep in sync, and the script's changes are visible in a ``git diff``
as a Python string. A dedicated integration test
(``tests/integration/test_rate_limit_lua.py``) runs the script against
a real Redis so a typo in a ``redis.call`` is caught before
production.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Final

from recsys.resilience.breaker import CircuitBreaker, CircuitOpenError

#: The Lua script.
#:
#: Called with:
#:   KEYS[1] = the bucket key
#:   ARGV[1] = capacity (bucket size in tokens)
#:   ARGV[2] = refill rate (tokens per second)
#:   ARGV[3] = now (unix seconds, fractional)
#:   ARGV[4] = cost (tokens this request consumes; normally 1)
#:
#: Returns a three-element array:
#:   {allowed (0 or 1), remaining (int), retry_after_seconds (int)}
#:
#: ``retry_after_seconds`` is 0 when allowed and the ceil of the deficit
#: divided by the refill rate otherwise.
#:
#: Algorithm version: v1 (token bucket).
_LUA_BUCKET_SCRIPT: Final[str] = r"""
-- rate_limit v1 (token bucket)
local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens') or ARGV[1])
local last   = tonumber(redis.call('HGET', KEYS[1], 'last')   or ARGV[3])
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local elapsed = now - last
if elapsed > 0 then
  tokens = math.min(capacity, tokens + elapsed * rate)
end

if tokens < cost then
  local deficit = cost - tokens
  local wait = math.ceil(deficit / rate)
  return {0, math.floor(tokens), wait}
end

tokens = tokens - cost
redis.call('HSET', KEYS[1], 'tokens', tokens, 'last', now)
redis.call('EXPIRE', KEYS[1], math.max(2, math.ceil(capacity / rate) * 2))
return {1, math.floor(tokens), 0}
"""


@dataclass(frozen=True)
class RateLimitDecision:
    """The outcome of one rate-limit check."""

    allowed: bool
    remaining: int
    retry_after_seconds: int
    degraded: bool


class _InMemoryBucket:
    """A single token bucket for the per-instance fallback."""

    __slots__ = ("last", "tokens")

    def __init__(self, capacity: float, now: float) -> None:
        self.tokens = capacity
        self.last = now


class TokenBucketLimiter:
    """Redis-backed token bucket with a per-instance fallback.

    The Redis path is server-side atomic and does not need a lock. The
    fallback dictionary is guarded by an asyncio lock so two concurrent
    requests cannot both read-modify-write the same in-memory bucket.

    The breaker is passed in, not built here: the same breaker guards
    the response cache (ADR-0015), and constructing it twice would
    defeat the point.
    """

    def __init__(
        self,
        *,
        redis_url: str,
        instance_count: int,
        bucket_seconds: float,
        breaker: CircuitBreaker,
    ) -> None:
        if instance_count < 1:
            raise ValueError(f"instance_count must be >= 1, got {instance_count}")
        if bucket_seconds <= 0:
            raise ValueError(f"bucket_seconds must be > 0, got {bucket_seconds}")

        self._redis_url = redis_url
        self._instance_count = instance_count
        self._bucket_seconds = bucket_seconds
        self._breaker = breaker

        self._redis: Any = None
        self._script: Any = None

        self._fallback: dict[str, _InMemoryBucket] = {}
        self._fallback_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def close(self) -> None:
        """Close the Redis connection. Safe to call more than once."""
        if self._redis is not None:
            # Best-effort: a broken connection's close is the OS's job.
            with contextlib.suppress(Exception):
                await self._redis.aclose()
            self._redis = None
            self._script = None

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    async def check(
        self,
        *,
        class_name: str,
        bucket_id: str,
        limit_per_minute: int,
    ) -> RateLimitDecision:
        """Return whether this request is allowed.

        ``class_name`` is the endpoint class (``recommend``, ``events``,
        ``admin``, ``ip``); it is part of the key. ``bucket_id`` is the
        identifier for the subject being limited: a credential hash for
        the credential bucket, an IP for the IP bucket.

        The Redis call is wrapped by the shared breaker. A refusal
        (breaker open, or a failed call) is not a request failure: the
        limiter falls back to its per-instance bucket and marks the
        decision ``degraded=True``.
        """
        if limit_per_minute < 1:
            raise ValueError(f"limit_per_minute must be >= 1, got {limit_per_minute}")

        capacity = self._bucket_capacity(limit_per_minute)
        refill_rate = limit_per_minute / 60.0
        key = f"{bucket_id}:{class_name}"
        now = time.time()

        # Fast path: an open breaker means "do not build the coroutine".
        if self._breaker.is_open():
            return await self._fallback_check(
                key=key, capacity=capacity, refill_rate=refill_rate, now=now
            )

        try:
            return await self._breaker.call(
                self._redis_check(key=key, capacity=capacity, refill_rate=refill_rate, now=now)
            )
        except CircuitOpenError:
            # The breaker flipped between is_open() and call() — either
            # a probe is in flight or a concurrent failure opened it.
            return await self._fallback_check(
                key=key, capacity=capacity, refill_rate=refill_rate, now=now
            )
        except Exception as exc:
            # A genuine failure (ConnectionError, TimeoutError, ...) —
            # the breaker has already recorded it. Any other exception
            # that reaches here is a caller bug; either way the request
            # must not fail because the rate limiter is degraded.
            from recsys.monitoring.metrics import RATE_LIMIT_LUA_ERRORS_TOTAL

            RATE_LIMIT_LUA_ERRORS_TOTAL.labels(type=type(exc).__name__).inc()
            return await self._fallback_check(
                key=key, capacity=capacity, refill_rate=refill_rate, now=now
            )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _bucket_capacity(self, limit_per_minute: int) -> float:
        """Bucket size in tokens: the limit's per-second rate times the
        configured burst window (``bucket_seconds``)."""
        return max(1.0, (limit_per_minute / 60.0) * self._bucket_seconds)

    async def _redis_check(
        self,
        *,
        key: str,
        capacity: float,
        refill_rate: float,
        now: float,
    ) -> RateLimitDecision:
        if self._redis is None:
            import redis.asyncio as aioredis

            # redis-py 5.x declares ``from_url`` with ``**kwargs: Any``;
            # mypy strict reports the call as untyped. The parameters
            # passed below are the ones redis-py documents; the cast is
            # not needed because the return value is stored on an
            # ``Any`` field.
            self._redis = aioredis.from_url(  # type: ignore[no-untyped-call]
                self._redis_url,
                decode_responses=False,
                socket_timeout=0.05,
                socket_connect_timeout=0.05,
            )
            self._script = self._redis.register_script(_LUA_BUCKET_SCRIPT)

        result = await self._script(
            keys=[f"rate_limit:{key}"],
            args=[capacity, refill_rate, now, 1.0],
        )
        return RateLimitDecision(
            allowed=bool(result[0]),
            remaining=int(result[1]),
            retry_after_seconds=int(result[2]),
            degraded=False,
        )

    async def _fallback_check(
        self,
        *,
        key: str,
        capacity: float,
        refill_rate: float,
        now: float,
    ) -> RateLimitDecision:
        """Per-instance token bucket with the limit split by
        ``instance_count``.

        The split is the whole point: keeping the full limit on each
        instance multiplies the effective aggregate by the instance
        count. The split is approximate (a caller who lands unevenly
        gets a little more or less), which is accepted in exchange for
        the availability the fallback buys. See ADR-0014.
        """
        split_refill = refill_rate / self._instance_count
        split_capacity = max(1.0, capacity / self._instance_count)

        async with self._fallback_lock:
            bucket = self._fallback.get(key)
            if bucket is None:
                bucket = _InMemoryBucket(split_capacity, now)
                self._fallback[key] = bucket

            elapsed = now - bucket.last
            if elapsed > 0:
                bucket.tokens = min(split_capacity, bucket.tokens + elapsed * split_refill)
                bucket.last = now

            if bucket.tokens < 1.0:
                deficit = 1.0 - bucket.tokens
                wait = math.ceil(deficit / split_refill) if split_refill > 0 else 60
                return RateLimitDecision(
                    allowed=False,
                    remaining=int(bucket.tokens),
                    retry_after_seconds=wait,
                    degraded=True,
                )

            bucket.tokens -= 1.0
            return RateLimitDecision(
                allowed=True,
                remaining=int(bucket.tokens),
                retry_after_seconds=0,
                degraded=True,
            )
