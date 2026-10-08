"""Unit tests for CacheStore, using a fake Redis.

These tests do not open a socket: the fake client implements the two
methods the store calls (``get`` and ``set``) and the store is
constructed with a breaker that never trips on its own. The
end-to-end behaviors (breaker recovery, socket timeout, pool
integrity) live in tests/integration/test_cache.py.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from recsys.cache.store import CacheLookup, CacheStore
from recsys.cache.ttl import NEGATIVE_SENTINEL
from recsys.resilience.breaker import CircuitBreaker


# ---------------------------------------------------------------- #
# fakes
# ---------------------------------------------------------------- #
class FakeRedis:
    """A tiny in-memory stand-in for ``redis.asyncio.Redis``."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.get_calls = 0
        self.set_calls = 0
        self.get_exc: BaseException | None = None
        self.set_exc: BaseException | None = None

    async def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        if self.get_exc is not None:
            raise self.get_exc
        return self.data.get(key)

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        self.set_calls += 1
        if self.set_exc is not None:
            raise self.set_exc
        self.data[key] = value

    async def aclose(self) -> None:
        pass


def _breaker(*, failure_threshold: int = 5, open_seconds: float = 60.0) -> CircuitBreaker:
    return CircuitBreaker(
        name="redis",
        failure_threshold=failure_threshold,
        open_seconds=open_seconds,
        open_max_seconds=open_seconds * 12,
    )


def _store(breaker: CircuitBreaker | None = None) -> tuple[CacheStore, FakeRedis]:
    fake = FakeRedis()
    store = CacheStore(
        redis_url="redis://unused",
        breaker=breaker or _breaker(),
        socket_timeout_seconds=0.1,
    )
    # Inject the fake client directly; _ensure_redis will see it and
    # skip creating a real one.
    store._redis = fake
    return store, fake


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------- #
# constructor
# ---------------------------------------------------------------- #
def test_socket_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="socket_timeout_seconds"):
        CacheStore(
            redis_url="redis://x",
            breaker=_breaker(),
            socket_timeout_seconds=0.0,
        )


# ---------------------------------------------------------------- #
# lookup: hit / miss
# ---------------------------------------------------------------- #
def test_lookup_miss_on_empty() -> None:
    store, fake = _store()
    result = _run(store.lookup("k"))
    assert result.outcome is CacheLookup.MISS
    assert result.items is None
    assert fake.get_calls == 1


def test_lookup_hit_returns_items() -> None:
    store, _ = _store()
    _run(store.store("k", [("i_1", 0.9), ("i_2", 0.4)], ttl_seconds=60))
    result = _run(store.lookup("k"))
    assert result.outcome is CacheLookup.HIT
    assert result.items == (("i_1", 0.9), ("i_2", 0.4))


def test_store_writes_serialized_payload() -> None:
    store, fake = _store()
    _run(store.store("k", [("i_1", 0.9)], ttl_seconds=60))
    assert fake.data["k"] == b'[["i_1",0.9]]'


# ---------------------------------------------------------------- #
# lookup: negative
# ---------------------------------------------------------------- #
def test_negative_hit() -> None:
    store, _ = _store()
    _run(store.store_negative("k", ttl_seconds=30))
    result = _run(store.lookup("k"))
    assert result.outcome is CacheLookup.NEGATIVE
    assert result.items is None


def test_negative_sentinel_is_bytes() -> None:
    store, fake = _store()
    _run(store.store_negative("k", ttl_seconds=30))
    assert fake.data["k"] == NEGATIVE_SENTINEL


# ---------------------------------------------------------------- #
# lookup: bypass
# ---------------------------------------------------------------- #
def test_breaker_open_bypasses_without_redis_call() -> None:
    breaker = _breaker(failure_threshold=1)
    store, fake = _store(breaker)

    # Trip the breaker.
    async def boom() -> None:
        raise ConnectionError("forced")

    with pytest.raises(ConnectionError):
        _run(breaker.call(boom()))
    assert breaker.is_open()

    result = _run(store.lookup("k"))
    assert result.outcome is CacheLookup.BYPASS
    assert fake.get_calls == 0


def test_store_bypasses_when_breaker_open() -> None:
    breaker = _breaker(failure_threshold=1)
    store, fake = _store(breaker)

    async def boom() -> None:
        raise ConnectionError("forced")

    with pytest.raises(ConnectionError):
        _run(breaker.call(boom()))

    ok = _run(store.store("k", [("i_1", 0.9)], ttl_seconds=60))
    assert ok is False
    assert fake.set_calls == 0


# ---------------------------------------------------------------- #
# lookup: error
# ---------------------------------------------------------------- #
def test_redis_error_on_get_reports_error() -> None:
    store, fake = _store()
    fake.get_exc = ConnectionError("down")
    result = _run(store.lookup("k"))
    assert result.outcome is CacheLookup.ERROR
    assert result.items is None


def test_corrupted_payload_reports_error() -> None:
    store, fake = _store()
    fake.data["k"] = b"not-json"
    result = _run(store.lookup("k"))
    assert result.outcome is CacheLookup.ERROR


def test_payload_not_a_list_reports_error() -> None:
    store, fake = _store()
    fake.data["k"] = b'{"oops": true}'
    result = _run(store.lookup("k"))
    assert result.outcome is CacheLookup.ERROR


def test_payload_entry_wrong_shape_reports_error() -> None:
    store, fake = _store()
    fake.data["k"] = b'[["only-one-element"]]'
    result = _run(store.lookup("k"))
    assert result.outcome is CacheLookup.ERROR


def test_write_error_returns_false_without_raising() -> None:
    store, fake = _store()
    fake.set_exc = ConnectionError("down")
    ok = _run(store.store("k", [("i_1", 0.9)], ttl_seconds=60))
    assert ok is False


def test_write_with_zero_ttl_returns_false() -> None:
    store, fake = _store()
    ok = _run(store.store("k", [("i_1", 0.9)], ttl_seconds=0))
    assert ok is False
    assert fake.set_calls == 0


# ---------------------------------------------------------------- #
# roundtrip
# ---------------------------------------------------------------- #
def test_roundtrip_preserves_items_and_order() -> None:
    store, _ = _store()
    items = [("i_1", 0.9), ("i_2", 0.4), ("i_3", 0.1)]
    _run(store.store("k", items, ttl_seconds=60))
    result = _run(store.lookup("k"))
    assert result.items == tuple(items)


# ---------------------------------------------------------------- #
# close
# ---------------------------------------------------------------- #
def test_close_is_idempotent() -> None:
    store, _ = _store()
    _run(store.close())
    _run(store.close())
