"""Unit tests for TokenBucketLimiter (fallback + cooldown paths).

The Redis path itself is exercised against a real Redis in
tests/integration/test_rate_limit_lua.py; here we cover the in-process
fallback and the cooldown state machine without touching the network.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from recsys.rate_limit.limiter import RateLimitDecision, TokenBucketLimiter


# ---------------------------------------------------------------- #
# constructor validation
# ---------------------------------------------------------------- #
def test_instance_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="instance_count"):
        TokenBucketLimiter(
            redis_url="redis://x",
            instance_count=0,
            bucket_seconds=60.0,
            cooldown_seconds=5.0,
        )


def test_bucket_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="bucket_seconds"):
        TokenBucketLimiter(
            redis_url="redis://x",
            instance_count=1,
            bucket_seconds=0.0,
            cooldown_seconds=5.0,
        )


def test_cooldown_seconds_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="cooldown_seconds"):
        TokenBucketLimiter(
            redis_url="redis://x",
            instance_count=1,
            bucket_seconds=60.0,
            cooldown_seconds=-1.0,
        )


def test_check_rejects_zero_limit() -> None:
    limiter = _limiter()
    with pytest.raises(ValueError, match="limit_per_minute"):
        asyncio.run(limiter.check(class_name="x", bucket_id="y", limit_per_minute=0))


# ---------------------------------------------------------------- #
# is_cooldown_active
# ---------------------------------------------------------------- #
def test_cooldown_initially_inactive() -> None:
    assert _limiter().is_cooldown_active() is False


def test_cooldown_active_after_enter() -> None:
    limiter = _limiter(cooldown_seconds=60.0)
    limiter._enter_cooldown(RuntimeError("x"))
    assert limiter.is_cooldown_active() is True


# ---------------------------------------------------------------- #
# fallback path (cooldown pre-set so Redis is never contacted)
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
# cooldown state machine (Redis raising)
# ---------------------------------------------------------------- #
def test_redis_failure_enters_cooldown_and_falls_back() -> None:
    limiter = _BoomLimiter(
        redis_url="redis://x",
        instance_count=1,
        bucket_seconds=60.0,
        cooldown_seconds=60.0,
    )
    d = asyncio.run(limiter.check(class_name="recommend", bucket_id="u", limit_per_minute=60))
    assert d.degraded is True
    assert d.allowed is True
    assert limiter.redis_calls == 1
    assert limiter.is_cooldown_active() is True


def test_cooldown_prevents_repeated_redis_probes() -> None:
    limiter = _BoomLimiter(
        redis_url="redis://x",
        instance_count=1,
        bucket_seconds=60.0,
        cooldown_seconds=60.0,
    )
    for _ in range(5):
        asyncio.run(limiter.check(class_name="recommend", bucket_id="u", limit_per_minute=60))
    # After the first failure, the cooldown window swallows the rest.
    assert limiter.redis_calls == 1


def test_cooldown_anchored_to_last_failure() -> None:
    """A second call inside the cooldown window must not extend the window
    and must not probe Redis again (this is the whole point of the design)."""
    limiter = _BoomLimiter(
        redis_url="redis://x",
        instance_count=1,
        bucket_seconds=60.0,
        cooldown_seconds=5.0,
    )
    asyncio.run(limiter.check(class_name="x", bucket_id="u", limit_per_minute=60))
    first_until = limiter._cooldown_until
    time.sleep(0.05)
    asyncio.run(limiter.check(class_name="x", bucket_id="u", limit_per_minute=60))
    assert limiter.redis_calls == 1
    assert limiter._cooldown_until == first_until


# ---------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------- #
def _limiter(
    *,
    instance_count: int = 1,
    bucket_seconds: float = 60.0,
    cooldown_seconds: float = 5.0,
) -> TokenBucketLimiter:
    return TokenBucketLimiter(
        redis_url="redis://unused",
        instance_count=instance_count,
        bucket_seconds=bucket_seconds,
        cooldown_seconds=cooldown_seconds,
    )


def _force_fallback(limiter: TokenBucketLimiter) -> TokenBucketLimiter:
    # Pre-set a long cooldown so the Redis path is never attempted.
    limiter._cooldown_until = time.time() + 3600.0
    return limiter


class _BoomLimiter(TokenBucketLimiter):
    """Limiter whose Redis path always raises, so the cooldown path runs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.redis_calls = 0

    async def _redis_check(self, **kwargs: Any) -> RateLimitDecision:
        self.redis_calls += 1
        raise ConnectionError("boom")
