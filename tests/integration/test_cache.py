"""Integration tests for the cache store against a real Redis.

The unit tests use a fake client; these run against a real Redis and
cover the behaviors the fake cannot: a real TTL that elapses, a
negative entry that survives a round trip, and a bypass while the
breaker is open.

Skipped unless ``RECSYS_TEST_REDIS_URL`` is set.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from recsys.cache import CacheLookup, CacheStore
from recsys.resilience import CircuitBreaker

pytestmark = [pytest.mark.integration]


@pytest.fixture
def redis_url() -> str:
    url = os.environ.get("RECSYS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set RECSYS_TEST_REDIS_URL to run redis integration tests")
    return url


def _key(prefix: str = "k") -> str:
    return f"cache:it:{prefix}:{uuid.uuid4().hex}"


def _store(redis_url: str) -> CacheStore:
    return CacheStore(
        redis_url=redis_url,
        breaker=CircuitBreaker(
            name="redis-test-cache",
            failure_threshold=1000,
            open_seconds=1.0,
            open_max_seconds=10.0,
        ),
        socket_timeout_seconds=0.5,
    )


# ---------------------------------------------------------------- #
# happy path
# ---------------------------------------------------------------- #
def test_roundtrip(redis_url: str) -> None:
    async def main() -> tuple[CacheLookup, tuple[tuple[object, ...], ...] | None]:
        store = _store(redis_url)
        key = _key()
        try:
            await store.store(key, [("i_1", 0.9), ("i_2", 0.4)], ttl_seconds=60)
            result = await store.lookup(key)
            return result.outcome, result.items
        finally:
            await store.close()

    outcome, items = asyncio.run(main())
    assert outcome is CacheLookup.HIT
    assert items == (("i_1", 0.9), ("i_2", 0.4))


def test_negative_roundtrip(redis_url: str) -> None:
    async def main() -> CacheLookup:
        store = _store(redis_url)
        key = _key()
        try:
            await store.store_negative(key, ttl_seconds=30)
            return (await store.lookup(key)).outcome
        finally:
            await store.close()

    assert asyncio.run(main()) is CacheLookup.NEGATIVE


def test_miss_on_unknown_key(redis_url: str) -> None:
    async def main() -> CacheLookup:
        store = _store(redis_url)
        try:
            return (await store.lookup(_key())).outcome
        finally:
            await store.close()

    assert asyncio.run(main()) is CacheLookup.MISS


# ---------------------------------------------------------------- #
# TTL: really elapses
# ---------------------------------------------------------------- #
def test_ttl_elapses(redis_url: str) -> None:
    async def main() -> tuple[CacheLookup, CacheLookup]:
        store = _store(redis_url)
        key = _key()
        try:
            await store.store(key, [("i_1", 0.9)], ttl_seconds=1)
            first = (await store.lookup(key)).outcome
            await asyncio.sleep(2.0)
            second = (await store.lookup(key)).outcome
            return first, second
        finally:
            await store.close()

    first, second = asyncio.run(main())
    assert first is CacheLookup.HIT
    assert second is CacheLookup.MISS, "entry outlived its TTL"


# ---------------------------------------------------------------- #
# bypass: breaker open
# ---------------------------------------------------------------- #
def test_bypass_when_breaker_open(redis_url: str) -> None:
    async def main() -> CacheLookup:
        breaker = CircuitBreaker(
            name="redis-test-bypass",
            failure_threshold=1,
            open_seconds=60.0,
            open_max_seconds=600.0,
        )

        async def boom() -> None:
            raise ConnectionError("forced")

        with pytest.raises(ConnectionError):
            await breaker.call(boom())

        store = CacheStore(
            redis_url=redis_url,
            breaker=breaker,
            socket_timeout_seconds=0.5,
        )
        try:
            return (await store.lookup(_key())).outcome
        finally:
            await store.close()

    assert asyncio.run(main()) is CacheLookup.BYPASS
