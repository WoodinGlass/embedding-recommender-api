"""Unit tests for the breaker-to-Prometheus callback (ADR-0015).

Metrics are read back through ``REGISTRY.get_sample_value`` (public API)
rather than the private ``_value`` attribute. That keeps mypy happy and
does not pin the tests to a prometheus_client internal. Each test uses a
unique breaker ``name`` so the process-wide counters cannot leak state
between tests; there is no reset helper to get wrong.
"""

from __future__ import annotations

import asyncio

import pytest

from recsys.monitoring.metrics import (
    BREAKER_CLOSED,
    BREAKER_HALF_OPEN,
    BREAKER_OPEN,
    REGISTRY,
)
from recsys.resilience import CircuitBreaker, CircuitState, TransitionEvent
from recsys.resilience.metrics import (
    breaker_transition_callback,
    set_initial_state,
)


# ---------------------------------------------------------------- #
# readers (public registry API)
# ---------------------------------------------------------------- #
def _gauge(name: str) -> float:
    value = REGISTRY.get_sample_value("recsys_circuit_breaker_state", {"name": name})
    assert value is not None, f"gauge not emitted for {name!r}"
    return value


def _changes(name: str, from_state: str, to_state: str) -> float:
    value = REGISTRY.get_sample_value(
        "recsys_circuit_breaker_state_changes_total",
        {"name": name, "from": from_state, "to": to_state},
    )
    return value or 0.0


def _trips(name: str, reason: str) -> float:
    value = REGISTRY.get_sample_value(
        "recsys_circuit_breaker_trips_total",
        {"name": name, "reason": reason},
    )
    return value or 0.0


async def _boom() -> None:
    raise ConnectionError("forced")


def _breaker(
    name: str,
    *,
    threshold: int = 1,
    open_seconds: float = 60.0,
    clock: object = None,
) -> CircuitBreaker:
    kwargs: dict[str, object] = {
        "name": name,
        "failure_threshold": threshold,
        "open_seconds": open_seconds,
        "open_max_seconds": open_seconds * 12,
        "on_transition": breaker_transition_callback,
    }
    if clock is not None:
        kwargs["clock"] = clock
    return CircuitBreaker(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------- #
# set_initial_state
# ---------------------------------------------------------------- #
def test_set_initial_state_closed() -> None:
    set_initial_state("t_initial_a", CircuitState.CLOSED)
    assert _gauge("t_initial_a") == BREAKER_CLOSED


def test_set_initial_state_open() -> None:
    set_initial_state("t_initial_b", CircuitState.OPEN)
    assert _gauge("t_initial_b") == BREAKER_OPEN


# ---------------------------------------------------------------- #
# callback: direct (no breaker)
# ---------------------------------------------------------------- #
def test_callback_sets_gauge_and_counters() -> None:
    name = "t_cb_a"
    breaker_transition_callback(
        TransitionEvent(
            name=name,
            from_state=CircuitState.CLOSED,
            to_state=CircuitState.OPEN,
            reason="threshold",
        )
    )
    assert _gauge(name) == BREAKER_OPEN
    assert _changes(name, "closed", "open") == 1
    assert _trips(name, "threshold") == 1


def test_callback_half_open_is_not_a_trip() -> None:
    name = "t_cb_b"
    breaker_transition_callback(
        TransitionEvent(
            name=name,
            from_state=CircuitState.OPEN,
            to_state=CircuitState.HALF_OPEN,
            reason="probe",
        )
    )
    assert _gauge(name) == BREAKER_HALF_OPEN
    assert _changes(name, "open", "half_open") == 1
    assert _trips(name, "threshold") == 0
    assert _trips(name, "probe_failed") == 0


def test_callback_close_after_probe_is_not_a_trip() -> None:
    name = "t_cb_c"
    breaker_transition_callback(
        TransitionEvent(
            name=name,
            from_state=CircuitState.HALF_OPEN,
            to_state=CircuitState.CLOSED,
            reason="probe_succeeded",
        )
    )
    assert _gauge(name) == BREAKER_CLOSED
    assert _changes(name, "half_open", "closed") == 1


def test_callback_probe_failed_counts_as_trip() -> None:
    name = "t_cb_d"
    breaker_transition_callback(
        TransitionEvent(
            name=name,
            from_state=CircuitState.HALF_OPEN,
            to_state=CircuitState.OPEN,
            reason="probe_failed",
        )
    )
    assert _gauge(name) == BREAKER_OPEN
    assert _trips(name, "probe_failed") == 1


# ---------------------------------------------------------------- #
# callback: end-to-end via a real breaker
# ---------------------------------------------------------------- #
def test_breaker_threshold_opens_and_emits_metrics() -> None:
    name = "t_e2e_a"
    b = _breaker(name, threshold=1)
    with pytest.raises(ConnectionError):
        asyncio.run(b.call(_boom()))
    assert b.state() is CircuitState.OPEN
    assert _gauge(name) == BREAKER_OPEN
    assert _changes(name, "closed", "open") == 1
    assert _trips(name, "threshold") == 1


def test_breaker_probe_success_returns_to_closed_gauge() -> None:
    name = "t_e2e_b"
    now = [1000.0]
    b = _breaker(name, threshold=1, open_seconds=1.0, clock=lambda: now[0])

    async def ok() -> str:
        return "ok"

    with pytest.raises(ConnectionError):
        asyncio.run(b.call(_boom()))
    assert _gauge(name) == BREAKER_OPEN

    now[0] += 1.0  # advance past the window; the next call is the probe
    assert asyncio.run(b.call(ok())) == "ok"
    assert b.state() is CircuitState.CLOSED
    assert _gauge(name) == BREAKER_CLOSED
    assert _changes(name, "half_open", "closed") == 1
