"""Integration tests for the Lua token bucket against a real Redis.

The point of this file is to catch a typo in the Lua script (a bad
``redis.call``, a wrong ARGV slot) before it reaches production. The
unit tests cannot catch that because they do not run the script.

Each test funnels every ``check()`` call through a single
``asyncio.run()``. Creating a new event loop per call reuses a Redis
client bound to a closed loop and turns every check after the first
into a degraded fallback, which is what the previous version of this
file did and why CI caught it and the unit tests did not.

Skipped unless ``RECSYS_TEST_REDIS_URL`` is set. CI provides it via a
``redis:7`` service container.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Coroutine
from typing import Any, TypeVar

import pytest

from recsys.rate_limit.limiter import RateLimitDecision, TokenBucketLimiter

pytestmark = [pytest.mark.integration]

T = TypeVar("T")


def _run(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on a fresh event loop, once per test."""
    return asyncio.run(coro)


async def _check(
    limiter: TokenBucketLimiter,
    *,
    class_name: str,
    bucket_id: str,
    limit_per_minute: int,
) -> RateLimitDecision:
    return await limiter.check(
        class_name=class_name,
        bucket_id=bucket_id,
        limit_per_minute=limit_per_minute,
    )


async def _exhaust(
    limiter: TokenBucketLimiter,
    *,
    class_name: str,
    bucket_id: str,
    limit_per_minute: int,
) -> list[RateLimitDecision]:
    """Take every token out of the bucket; return the decisions."""
    out: list[RateLimitDecision] = []
    for _ in range(limit_per_minute):
        out.append(
            await _check(
                limiter,
                class_name=class_name,
                bucket_id=bucket_id,
                limit_per_minute=limit_per_minute,
            )
        )
    return out


@pytest.fixture
def redis_url() -> str:
    url = os.environ.get("RECSYS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set RECSYS_TEST_REDIS_URL to run redis integration tests")
    return url


@pytest.fixture
def limiter(redis_url: str) -> TokenBucketLimiter:
    return TokenBucketLimiter(
        redis_url=redis_url,
        instance_count=1,
        bucket_seconds=60.0,
        cooldown_seconds=5.0,
    )


def _unique_bucket() -> str:
    return f"test-{uuid.uuid4().hex}"


def test_lua_allows_first_request(limiter: TokenBucketLimiter) -> None:
    async def _go() -> RateLimitDecision:
        return await _check(
            limiter,
            class_name="recommend",
            bucket_id=_unique_bucket(),
            limit_per_minute=60,
        )

    d = _run(_go())
    assert d.allowed is True
    assert d.degraded is False
    assert d.retry_after_seconds == 0


def test_lua_denies_when_exhausted(limiter: TokenBucketLimiter) -> None:
    async def _go() -> tuple[list[RateLimitDecision], RateLimitDecision]:
        bucket = _unique_bucket()
        first = await _exhaust(
            limiter,
            class_name="recommend",
            bucket_id=bucket,
            limit_per_minute=60,
        )
        last = await _check(
            limiter,
            class_name="recommend",
            bucket_id=bucket,
            limit_per_minute=60,
        )
        return first, last

    first, last = _run(_go())
    assert all(d.allowed for d in first), "all tokens should be granted"
    assert all(d.degraded is False for d in first), "Lua path should be used"
    assert last.allowed is False
    assert last.degraded is False
    assert last.retry_after_seconds >= 1


def test_lua_buckets_are_independent(limiter: TokenBucketLimiter) -> None:
    async def _go() -> tuple[RateLimitDecision, RateLimitDecision]:
        b1 = _unique_bucket()
        b2 = _unique_bucket()
        await _exhaust(limiter, class_name="recommend", bucket_id=b1, limit_per_minute=60)
        d1 = await _check(limiter, class_name="recommend", bucket_id=b1, limit_per_minute=60)
        d2 = await _check(limiter, class_name="recommend", bucket_id=b2, limit_per_minute=60)
        return d1, d2

    d1, d2 = _run(_go())
    assert d1.allowed is False
    assert d2.allowed is True


def test_lua_classes_are_independent(limiter: TokenBucketLimiter) -> None:
    async def _go() -> tuple[RateLimitDecision, RateLimitDecision]:
        bucket = _unique_bucket()
        await _exhaust(limiter, class_name="recommend", bucket_id=bucket, limit_per_minute=60)
        d_rec = await _check(limiter, class_name="recommend", bucket_id=bucket, limit_per_minute=60)
        d_ip = await _check(limiter, class_name="ip", bucket_id=bucket, limit_per_minute=60)
        return d_rec, d_ip

    d_rec, d_ip = _run(_go())
    assert d_rec.allowed is False
    assert d_ip.allowed is True
