"""Unit tests for the circuit breaker (ADR-0015)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from recsys.resilience.breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    TransitionEvent,
)


# ---------------------------------------------------------------- #
# clock + helpers
# ---------------------------------------------------------------- #
class FakeClock:
    """Monotonic-like clock the test controls."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _breaker(
    clock: FakeClock | None = None,
    *,
    name: str = "redis",
    failure_threshold: int = 3,
    open_seconds: float = 5.0,
    open_max_seconds: float = 60.0,
    on_transition: Any = None,
) -> CircuitBreaker:
    return CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        open_seconds=open_seconds,
        open_max_seconds=open_max_seconds,
        clock=clock or FakeClock(),
        on_transition=on_transition,
    )


async def _ok() -> str:
    return "ok"


async def _boom(exc: BaseException) -> Any:
    raise exc


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _refuse(b: CircuitBreaker) -> CircuitOpenError:
    """Call a breaker that should be open; return the refusal.

    The inner ``_ok()`` coroutine is created eagerly and, because the
    breaker raises before awaiting, would emit a "coroutine was never
    awaited" warning at GC time. ``coro.close()`` in the ``finally``
    silences it.
    """
    coro = _ok()
    try:
        with pytest.raises(CircuitOpenError) as ei:
            _run(b.call(coro))
    finally:
        coro.close()
    return ei.value


# ---------------------------------------------------------------- #
# constructor validation
# ---------------------------------------------------------------- #
def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        _breaker(failure_threshold=0)


def test_open_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="open_seconds"):
        _breaker(open_seconds=0)


def test_open_max_must_be_at_least_open() -> None:
    with pytest.raises(ValueError, match="open_max_seconds"):
        _breaker(open_seconds=10.0, open_max_seconds=5.0)


# ---------------------------------------------------------------- #
# initial state
# ---------------------------------------------------------------- #
def test_starts_closed() -> None:
    b = _breaker()
    assert b.state() is CircuitState.CLOSED
    assert b.is_open() is False


# ---------------------------------------------------------------- #
# success path
# ---------------------------------------------------------------- #
def test_call_returns_value_on_success() -> None:
    b = _breaker()
    assert _run(b.call(_ok())) == "ok"
    assert b.state() is CircuitState.CLOSED


def test_success_resets_consecutive_failure_count() -> None:
    clock = FakeClock()
    b = _breaker(clock, failure_threshold=3)
    # two failures, then a success, then two more: still under threshold
    for _ in range(2):
        with pytest.raises(ConnectionError):
            _run(b.call(_boom(ConnectionError("down"))))
    assert _run(b.call(_ok())) == "ok"
    for _ in range(2):
        with pytest.raises(ConnectionError):
            _run(b.call(_boom(ConnectionError("down"))))
    assert b.state() is CircuitState.CLOSED


# ---------------------------------------------------------------- #
# failure -> open
# ---------------------------------------------------------------- #
def test_threshold_failures_open_breaker() -> None:
    b = _breaker(failure_threshold=3)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            _run(b.call(_boom(ConnectionError("down"))))
    assert b.state() is CircuitState.OPEN
    assert b.is_open() is True


def test_open_breaker_refuses_call_without_awaiting() -> None:
    b = _breaker(failure_threshold=1)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    exc = _refuse(b)
    assert exc.reason == "open"
    assert "redis" in str(exc)


def test_is_open_false_after_timer_elapsed() -> None:
    """After the OPEN window elapses, ``is_open()`` reports False.

    Otherwise a caller that fast-paths on ``is_open()`` (the cache and
    the limiter both do) would bypass the probe for as long as it kept
    asking, and a dependency that recovered would never be re-tried.
    This is the bug the integration test
    ``test_threshold_failures_then_recovery`` caught.
    """
    clock = FakeClock()
    b = _breaker(clock, failure_threshold=1, open_seconds=5.0)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    assert b.is_open() is True
    clock.advance(5.0)
    assert b.is_open() is False
    # And the next call is admitted, closing the breaker.
    assert _run(b.call(_ok())) == "ok"
    assert b.state() is CircuitState.CLOSED


# ---------------------------------------------------------------- #
# failure classification (K-A)
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    "exc",
    [
        ConnectionError("x"),
        TimeoutError("x"),
    ],
)
def test_builtin_failure_classes_count(exc: BaseException) -> None:
    b = _breaker(failure_threshold=1)
    with pytest.raises(type(exc)):
        _run(b.call(_boom(exc)))
    assert b.state() is CircuitState.OPEN


def test_named_failure_classes_count() -> None:
    # Fake the class name without importing redis or psycopg.
    class OperationalError(Exception):
        pass

    b = _breaker(failure_threshold=1)
    with pytest.raises(OperationalError):
        _run(b.call(_boom(OperationalError("server gone"))))
    assert b.state() is CircuitState.OPEN


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("caller bug"),
        TypeError("caller bug"),
        KeyError("caller bug"),
    ],
)
def test_logic_errors_do_not_count(exc: BaseException) -> None:
    b = _breaker(failure_threshold=1)
    with pytest.raises(type(exc)):
        _run(b.call(_boom(exc)))
    # Breaker unchanged: a caller bug is not a dependency outage.
    assert b.state() is CircuitState.CLOSED
    assert b.is_open() is False


def test_named_non_failure_classes_do_not_count() -> None:
    class ResponseError(Exception):
        """Same name as redis.ResponseError; deliberately not a failure."""

    b = _breaker(failure_threshold=1)
    with pytest.raises(ResponseError):
        _run(b.call(_boom(ResponseError("bad reply"))))
    assert b.state() is CircuitState.CLOSED


# ---------------------------------------------------------------- #
# HALF_OPEN probe
# ---------------------------------------------------------------- #
def test_open_moves_to_half_open_after_timer() -> None:
    clock = FakeClock()
    b = _breaker(clock, failure_threshold=1, open_seconds=5.0)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    assert b.state() is CircuitState.OPEN
    clock.advance(5.0)
    # The first call after the timer runs the probe.
    assert _run(b.call(_ok())) == "ok"
    assert b.state() is CircuitState.CLOSED


def test_half_open_probe_failure_reopens_with_backoff() -> None:
    clock = FakeClock()
    b = _breaker(clock, failure_threshold=1, open_seconds=5.0, open_max_seconds=60.0)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    clock.advance(5.0)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("still down"))))
    assert b.state() is CircuitState.OPEN
    # Immediate retry is refused (new window has not elapsed).
    _refuse(b)
    # After the doubled window, a probe is admitted.
    clock.advance(10.0)
    assert _run(b.call(_ok())) == "ok"
    assert b.state() is CircuitState.CLOSED


def test_backoff_caps_at_open_max() -> None:
    clock = FakeClock()
    b = _breaker(clock, failure_threshold=1, open_seconds=5.0, open_max_seconds=12.0)

    # Trip 1: threshold reached -> OPEN with the initial window (5s).
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    assert b._current_open_seconds == 5.0

    # Failed probe -> window doubles to 10s.
    clock.advance(5.0)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    assert b._current_open_seconds == 10.0

    # Failed probe -> min(20, 12) = 12s (capped).
    clock.advance(10.0)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    assert b._current_open_seconds == 12.0

    # Failed probe -> min(24, 12) = 12s (still capped).
    clock.advance(12.0)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    assert b._current_open_seconds == 12.0

    # 11s < 12s: still refused.
    clock.advance(11.0)
    _refuse(b)

    # After the full 12s window, a probe is admitted and closes on success.
    clock.advance(1.0)
    assert _run(b.call(_ok())) == "ok"
    assert b.state() is CircuitState.CLOSED


# ---------------------------------------------------------------- #
# single probe in HALF_OPEN
# ---------------------------------------------------------------- #
def test_single_probe_when_half_open() -> None:
    clock = FakeClock()

    async def main() -> str:
        b = _breaker(clock, failure_threshold=1, open_seconds=1.0)
        with pytest.raises(ConnectionError):
            await b.call(_boom(ConnectionError("down")))
        clock.advance(1.0)

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_probe() -> str:
            started.set()
            await release.wait()
            return "ok"

        # First call becomes the probe; second sees HALF_OPEN.
        # The inner `_ok()` coroutine is closed explicitly: the breaker
        # raises before awaiting, and without close() the coroutine
        # leaks a "never awaited" warning at GC time.
        probe = asyncio.create_task(b.call(slow_probe()))
        await started.wait()
        second = _ok()
        try:
            with pytest.raises(CircuitOpenError) as ei:
                await b.call(second)
        finally:
            second.close()
        assert ei.value.reason == "probe_in_flight"
        release.set()
        return await probe

    result = asyncio.run(main())
    assert result == "ok"


# ---------------------------------------------------------------- #
# transition callback
# ---------------------------------------------------------------- #
def test_transition_callback_receives_events() -> None:
    events: list[TransitionEvent] = []
    b = _breaker(
        failure_threshold=1,
        open_seconds=1.0,
        on_transition=events.append,
    )
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    assert len(events) == 1
    ev = events[0]
    assert ev.name == "redis"
    assert ev.from_state is CircuitState.CLOSED
    assert ev.to_state is CircuitState.OPEN
    assert ev.reason == "threshold"


def test_callback_exception_does_not_corrupt_state() -> None:
    def boom(_ev: TransitionEvent) -> None:
        raise RuntimeError("callback bug")

    b = _breaker(failure_threshold=1, on_transition=boom)
    with pytest.raises(ConnectionError):
        _run(b.call(_boom(ConnectionError("down"))))
    # State still moved despite the callback raising.
    assert b.state() is CircuitState.OPEN
