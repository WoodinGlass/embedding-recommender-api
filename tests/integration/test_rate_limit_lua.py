"""Integration tests for the Lua token bucket against a real Redis.

The point of this file is to catch a typo in the Lua script (a bad
``redis.call``, a wrong ARGV slot) before it reaches production. The unit
tests cannot catch that because they do not run the script.

Skipped unless ``RECSYS_TEST_REDIS_URL`` is set. CI provides it via a
``redis:7`` service container.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from recsys.rate_limit.limiter import TokenBucketLimiter

pytestmark = [pytest.mark.integration]


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
    d = asyncio.run(
        limiter.check(class_name="recommend", bucket_id=_unique_bucket(), limit_per_minute=60)
    )
    assert d.allowed is True
    assert d.degraded is False
    assert d.retry_after_seconds == 0


def test_lua_denies_when_exhausted(limiter: TokenBucketLimiter) -> None:
    bucket = _unique_bucket()
    for _ in range(60):
        d = asyncio.run(
            limiter.check(class_name="recommend", bucket_id=bucket, limit_per_minute=60)
        )
        assert d.allowed is True
    d = asyncio.run(limiter.check(class_name="recommend", bucket_id=bucket, limit_per_minute=60))
    assert d.allowed is False
    assert d.retry_after_seconds >= 1


def test_lua_buckets_are_independent(limiter: TokenBucketLimiter) -> None:
    b1 = _unique_bucket()
    b2 = _unique_bucket()
    for _ in range(60):
        asyncio.run(limiter.check(class_name="recommend", bucket_id=b1, limit_per_minute=60))
    d1 = asyncio.run(limiter.check(class_name="recommend", bucket_id=b1, limit_per_minute=60))
    d2 = asyncio.run(limiter.check(class_name="recommend", bucket_id=b2, limit_per_minute=60))
    assert d1.allowed is False
    assert d2.allowed is True


def test_lua_classes_are_independent(limiter: TokenBucketLimiter) -> None:
    bucket = _unique_bucket()
    for _ in range(60):
        asyncio.run(limiter.check(class_name="recommend", bucket_id=bucket, limit_per_minute=60))
    d_rec = asyncio.run(
        limiter.check(class_name="recommend", bucket_id=bucket, limit_per_minute=60)
    )
    d_ip = asyncio.run(limiter.check(class_name="ip", bucket_id=bucket, limit_per_minute=60))
    assert d_rec.allowed is False
    assert d_ip.allowed is True
