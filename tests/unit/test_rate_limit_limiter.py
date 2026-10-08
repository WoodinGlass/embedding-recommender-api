"""Unit tests for TokenBucketLimiter (fallback + breaker paths).

The Redis path itself is exercised against a real Redis in
tests/integration/test_rate_limit_lua.py; here we cover the in-process
fallback and the shared breaker without touching the network.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from recsys.rate_limit.limiter import RateLimitDecision, TokenBucketLimiter
from recsys.resilience.breaker import CircuitBreaker


# ---------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------- #
def _breaker(
    *,
    failure_threshold: int = 1,
    open_seconds: float = 60.0,
) -> CircuitBreaker:
    return CircuitBreaker(
        name="redis",
        failure_threshold=failure_threshold,
        open_seconds=open_seconds,
        open_max_seconds=open_seconds * 12,
    )


def _limiter(
    *,
    instance_count: int = 1,
    bucket_seconds: float = 60.0,
    breaker: CircuitBreaker | None = None,
) -> TokenBucketLimiter:
    return TokenBucketLimiter(
        redis_url="redis://unused",
        instance_count=instance_count,
        bucket_seconds=bucket_seconds,
        breaker=breaker or _breaker(),
    )


async def _boom_conn() -> None:
    raise ConnectionError("forced")


def _force_fallback(limiter: TokenBucketLimiter) -> TokenBucketLimiter:
    """Trip the limiter's breaker so the fallback path is active.

    Pre-tripping the breaker is the M3.3 equivalent of M3.2's
    "pre-set cooldown": it short-circuits the Redis path so a unit
    test never opens a socket. The ``ConnectionError`` the probe
    raises is the signal the breaker records; it is suppressed here
    because the test cares about the post-condition, not the probe.
    """
    with contextlib.suppress(ConnectionError):
        asyncio.run(limiter._breaker.call(_boom_conn()))
    assert limiter._breaker.is_open()
    return limiter


class _BoomLimiter(TokenBucketLimiter):
    """Limiter whose Redis path always raises, to exercise the breaker."""

    def __init__(self, *, breaker: CircuitBreaker, **kwargs: Any) -> None:
        super().__init__(breaker=breaker, **kwargs)
        self.redis_calls = 0

    async def _redis_check(self, **kwargs: Any) -> RateLimitDecision:
        self.redis_calls += 1
        raise ConnectionError("boom")


# ---------------------------------------------------------------- #
# constructor validation
# ---------------------------------------------------------------- #
def test_instance_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="instance_count"):
        TokenBucketLimiter(
            redis_url="redis://x",
            instance_count=0,
            bucket_seconds=60.0,
            breaker=_breaker(),
        )


def test_bucket_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="bucket_seconds"):
        TokenBucketLimiter(
            redis_url="redis://x",
            instance_count=1,
            bucket_seconds=0.0,
            breaker=_breaker(),
        )


def test_check_rejects_zero_limit() -> None:
    limiter = _limiter()
    with pytest.raises(ValueError, match="limit_per_minute"):
        asyncio.run(limiter.check(class_name="x", bucket_id="y", limit_per_minute=0))


# ---------------------------------------------------------------- #
# fallback path (breaker pre-tripped, so Redis is never contacted)
# ---------------------------------------------------------------- #
def test_fallback_allows_first_request() -> None:
    limiter = _force_fallback(_limiter(instance_count=1))
    d = asyncio.run(limiter.check(class_name="recommend", bucket_id="u1", limit_per_minute=60))
    assert isinstance(d, RateLimitDecision)
    assert d.allowed is True
    assert d.degraded is True
    assert d.retry_after_seconds == 0


def test_fallback_denies_when_bucket_empty() -> None:
    # limit=60/min, bucket_seconds=60 -> capacity 60 tokens, instance_count=1.
    limiter = _force_fallback(_limiter(instance_count=1, bucket_seconds=60.0))
    for _ in range(60):
        d = asyncio.run(limiter.check(class_name="recommend", bucket_id="u1", limit_per_minute=60))
        assert d.allowed is True
    d = asyncio.run(limiter.check(class_name="recommend", bucket_id="u1", limit_per_minute=60))
    assert d.allowed is False
    assert d.retry_after_seconds >= 1


def test_fallback_split_by_instance_count() -> None:
    # limit=60/min, bucket 60s -> capacity 60; split by 4 -> 15 tokens.
    limiter = _force_fallback(_limiter(instance_count=4, bucket_seconds=60.0))
    allowed = 0
    for _ in range(20):
        d = asyncio.run(limiter.check(class_name="recommend", bucket_id="u1", limit_per_minute=60))
        if d.allowed:
            allowed += 1
        else:
            break
    assert allowed == 15


def test_fallback_keys_are_independent() -> None:
    limiter = _force_fallback(_limiter(instance_count=1, bucket_seconds=60.0))
    for _ in range(60):
        asyncio.run(limiter.check(class_name="recommend", bucket_id="u1", limit_per_minute=60))
    d_u1 = asyncio.run(limiter.check(class_name="recommend", bucket_id="u1", limit_per_minute=60))
    d_u2 = asyncio.run(limiter.check(class_name="recommend", bucket_id="u2", limit_per_minute=60))
    assert d_u1.allowed is False
    assert d_u2.allowed is True


def test_fallback_classes_are_independent() -> None:
    limiter = _force_fallback(_limiter(instance_count=1, bucket_seconds=60.0))
    for _ in range(60):
        asyncio.run(limiter.check(class_name="recommend", bucket_id="u", limit_per_minute=60))
    d_rec = asyncio.run(limiter.check(class_name="recommend", bucket_id="u", limit_per_minute=60))
    d_ip = asyncio.run(limiter.check(class_name="ip", bucket_id="u", limit_per_minute=60))
    assert d_rec.allowed is False
    assert d_ip.allowed is True


def test_fallback_refills_over_time() -> None:
    # bucket_seconds=1 -> capacity = 60/60 * 1 = 1 token for limit=60.
    limiter = _force_fallback(_limiter(instance_count=1, bucket_seconds=1.0))
    d1 = asyncio.run(limiter.check(class_name="x", bucket_id="u", limit_per_minute=60))
    d2 = asyncio.run(limiter.check(class_name="x", bucket_id="u", limit_per_minute=60))
    assert d1.allowed is True
    assert d2.allowed is False
    # Rewind the fallback bucket's clock by 2 seconds; refill should allow one more.
    bucket = limiter._fallback["u:x"]
    bucket.last -= 2.0
    d3 = asyncio.run(limiter.check(class_name="x", bucket_id="u", limit_per_minute=60))
    assert d3.allowed is True


# ---------------------------------------------------------------- #
# breaker integration (Redis raising)
# ---------------------------------------------------------------- #
def test_redis_failure_opens_breaker_and_falls_back() -> None:
    breaker = _breaker(failure_threshold=1, open_seconds=60.0)
    limiter = _BoomLimiter(
        breaker=breaker,
        redis_url="redis://x",
        instance_count=1,
        bucket_seconds=60.0,
    )
    d = asyncio.run(limiter.check(class_name="recommend", bucket_id="u", limit_per_minute=60))
    assert d.degraded is True
    assert d.allowed is True
    assert limiter.redis_calls == 1
    assert breaker.is_open() is True


def test_open_breaker_prevents_repeated_redis_probes() -> None:
    breaker = _breaker(failure_threshold=1, open_seconds=60.0)
    limiter = _BoomLimiter(
        breaker=breaker,
        redis_url="redis://x",
        instance_count=1,
        bucket_seconds=60.0,
    )
    for _ in range(5):
        asyncio.run(limiter.check(class_name="recommend", bucket_id="u", limit_per_minute=60))
    # After the first failure opened the breaker, the fast path in
    # check() skips _redis_check entirely.
    assert limiter.redis_calls == 1


def test_logic_error_does_not_open_breaker() -> None:
    """A non-failure exception (a caller bug) must not trip the breaker.

    The limiter still falls back — a rate limiter must not fail a
    request because the limiter itself has a bug — but the shared
    breaker stays closed, so the cache path is not affected.
    """

    class LogicBoom(TokenBucketLimiter):
        async def _redis_check(self, **kwargs: Any) -> RateLimitDecision:
            raise ValueError("caller bug")

    breaker = _breaker(failure_threshold=1, open_seconds=60.0)
    limiter = LogicBoom(
        breaker=breaker,
        redis_url="redis://x",
        instance_count=1,
        bucket_seconds=60.0,
    )
    d = asyncio.run(limiter.check(class_name="recommend", bucket_id="u", limit_per_minute=60))
    assert d.degraded is True
    assert d.allowed is True
    assert breaker.is_open() is False
