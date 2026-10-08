"""Integration tests for the shared breaker against a real Redis.

Two things the unit tests cannot show:

- **Socket timeout does not leave a poisoned connection pool.** The
  timeout lives in ``redis-py`` (``socket_timeout``), not in
  ``asyncio.wait_for`` (ADR-0015 D5). A cancelled ``wait_for`` leaves
  a redis-py connection in a state the pool does not always recover
  cleanly; ``socket_timeout`` is handled inside redis-py. The proof
  is a burst of real timeouts followed by a request that must
  succeed.

- **A logic error in one shared path does not open the breaker for
  the other.** The cache and the limiter share one Redis breaker;
  the failure classification (K-A) means a ``ValueError`` from a
  caller bug does not count. The proof is a shared breaker that
  stays CLOSED after a logic error, so the cache path keeps talking
  to Redis.

Skipped unless ``RECSYS_TEST_REDIS_URL`` is set; CI provides a
``redis:7`` service container.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from recsys.cache import CacheStore
from recsys.resilience import CircuitBreaker, CircuitState

# Redis is an optional dependency ([cache] extra). When it is not
# installed the module skips cleanly, instead of failing collection
# with an ImportError that masks the reason for the skip. This
# runtime call cannot sit between imports (E402), so it comes after
# every import statement, right before the module-level pytestmark.
_redis_exceptions = pytest.importorskip("redis.exceptions")
RedisError = _redis_exceptions.RedisError

pytestmark = [pytest.mark.integration]


@pytest.fixture
def redis_url() -> str:
    url = os.environ.get("RECSYS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set RECSYS_TEST_REDIS_URL to run redis integration tests")
    return url


def _fresh_key(prefix: str) -> str:
    return f"it-{prefix}-{uuid.uuid4().hex}"


# ---------------------------------------------------------------- #
# D5: socket timeout does not poison the pool
# ---------------------------------------------------------------- #
def test_socket_timeout_leaves_pool_usable(redis_url: str) -> None:
    """A burst of socket timeouts must not break the next request.

    ``BLPOP`` on a missing key blocks for the given timeout; with a
    client ``socket_timeout`` well below it, every call raises
    ``TimeoutError`` after the client-side deadline. After that
    burst, a normal ``lookup`` must return MISS (not ERROR): the
    pool has to hand out a usable connection again.
    """

    async def main() -> tuple[int, str]:
        store = CacheStore(
            redis_url=redis_url,
            breaker=CircuitBreaker(
                name="redis-test-timeout",
                failure_threshold=1000,  # do not let the breaker open
                open_seconds=1.0,
                open_max_seconds=10.0,
            ),
            socket_timeout_seconds=0.05,
        )
        try:
            await store._ensure_redis()
            # 10 fast timeouts: cheap in CI, enough to prove the pool
            # recovers. Every call must raise a timeout, not hang.
            timeouts = 0
            for _ in range(10):
                with pytest.raises(RedisError):
                    # BLPOP blocks 1s server-side; the client gives up
                    # after 50 ms. Any exception class is accepted:
                    # redis-py has raised both ``TimeoutError`` and
                    # ``ConnectionError`` on different versions.
                    await store._redis.blpop([_fresh_key("q")], timeout=1)
                timeouts += 1

            result = await store.lookup(_fresh_key("k"))
            return timeouts, result.outcome.value
        finally:
            await store.close()

    timeouts, outcome = asyncio.run(main())
    assert timeouts == 10
    assert outcome == "miss", f"pool poisoned: lookup returned {outcome!r}"


# ---------------------------------------------------------------- #
# K-A: logic error does not open the shared breaker
# ---------------------------------------------------------------- #
def test_logic_error_does_not_open_shared_breaker(redis_url: str) -> None:
    """A caller bug on one path must not open the breaker for the other.

    The scenario: one breaker is shared by a cache store and a
    hypothetical second path. The second path raises ``ValueError``
    (a caller bug, not a dependency failure). The breaker stays
    CLOSED, so the cache still talks to Redis rather than bypassing.
    """

    async def main() -> tuple[CircuitState, str]:
        breaker = CircuitBreaker(
            name="redis-test-shared",
            failure_threshold=1,  # one failure would open it — if it counted
            open_seconds=60.0,
            open_max_seconds=600.0,
        )
        store = CacheStore(
            redis_url=redis_url,
            breaker=breaker,
            socket_timeout_seconds=0.5,
        )
        try:

            async def caller_bug() -> None:
                raise ValueError("a logic error, not a connection error")

            with pytest.raises(ValueError, match="logic error"):
                await breaker.call(caller_bug())

            state = breaker.state()
            result = await store.lookup(_fresh_key("k"))
            return state, result.outcome.value
        finally:
            await store.close()

    state, outcome = asyncio.run(main())
    assert state is CircuitState.CLOSED, "logic error opened the shared breaker"
    assert outcome == "miss", f"cache bypassed after a logic error: {outcome!r}"


# ---------------------------------------------------------------- #
# End-to-end: threshold failures open; probe success closes
# ---------------------------------------------------------------- #
def test_threshold_failures_then_recovery(redis_url: str) -> None:
    """Trip the breaker with real connection failures, then recover.

    The failing calls connect to a closed port (a real
    ``ConnectionError`` from the OS), and the recovering call uses
    the live ``redis_url``. A single breaker sees both.
    """

    async def main() -> tuple[CircuitState, CircuitState, str]:
        breaker = CircuitBreaker(
            name="redis-test-recover",
            failure_threshold=2,
            open_seconds=0.1,  # short: the probe must be admitted quickly
            open_max_seconds=1.0,
        )

        # A URL that nothing listens on. 127.0.0.1:1 is port 1
        # (traditionally tcpmux, unbound on every CI runner).
        dead_store = CacheStore(
            redis_url="redis://127.0.0.1:1/0",
            breaker=breaker,
            socket_timeout_seconds=0.2,
        )
        try:
            for _ in range(2):
                await dead_store.lookup(_fresh_key("dead"))
        finally:
            await dead_store.close()

        tripped = breaker.state()

        # Wait out the OPEN window, then let the live store run the
        # single probe. It must succeed and close the breaker.
        await asyncio.sleep(0.2)

        live_store = CacheStore(
            redis_url=redis_url,
            breaker=breaker,
            socket_timeout_seconds=0.5,
        )
        try:
            result = await live_store.lookup(_fresh_key("alive"))
        finally:
            await live_store.close()

        return tripped, breaker.state(), result.outcome.value

    tripped, recovered, outcome = asyncio.run(main())
    assert tripped is CircuitState.OPEN, "threshold did not open the breaker"
    assert recovered is CircuitState.CLOSED, "probe did not close the breaker"
    assert outcome == "miss"
