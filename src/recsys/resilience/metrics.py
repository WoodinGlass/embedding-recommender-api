"""Prometheus wiring for circuit-breaker transitions (ADR-0015).

The breaker itself does not import ``monitoring``: it reports state
transitions through an optional callback, and this module is the
callback that turns those events into metrics. Keeping the coupling
in one small file means:

- ``resilience/breaker.py`` stays testable without the Prometheus
  registry.
- The composition root (``api/app.py``) wires the two together with
  a single argument (``on_transition=breaker_transition_callback``).
- The metric semantics (which counter, which labels) live next to
  each other, in a file a reviewer can read in one screen.
"""

from __future__ import annotations

from recsys.monitoring.metrics import (
    BREAKER_CLOSED,
    BREAKER_HALF_OPEN,
    BREAKER_OPEN,
    CIRCUIT_BREAKER_STATE,
    CIRCUIT_BREAKER_STATE_CHANGES_TOTAL,
    CIRCUIT_BREAKER_TRIPS_TOTAL,
)
from recsys.resilience.breaker import CircuitState, TransitionEvent

#: Numeric value the gauge reports for each state. The constants come
#: from ``monitoring.metrics``; the mapping lives here so a state that
#: is added later surfaces as a ``KeyError`` at startup rather than a
#: silent wrong gauge value.
_STATE_VALUE: dict[CircuitState, int] = {
    CircuitState.CLOSED: BREAKER_CLOSED,
    CircuitState.HALF_OPEN: BREAKER_HALF_OPEN,
    CircuitState.OPEN: BREAKER_OPEN,
}


def set_initial_state(name: str, state: CircuitState) -> None:
    """Set the state gauge without emitting a transition.

    The breaker starts in CLOSED; without this call the gauge would
    be absent from ``/metrics`` until the first transition, which
    makes an alert on ``state != 0`` ambiguous between "closed" and
    "metric never emitted". Called once per breaker at startup.
    """
    CIRCUIT_BREAKER_STATE.labels(name=name).set(_STATE_VALUE[state])


def breaker_transition_callback(event: TransitionEvent) -> None:
    """Emit the three breaker metrics for a single transition.

    Passed to ``CircuitBreaker(on_transition=...)``. The callback is
    a pure function of ``event`` plus the process-wide Prometheus
    registry; it holds no state of its own.

    The ``state_changes_total`` counter is labelled with the *from*
    and *to* states (both bounded enums) so an alert can distinguish
    "the breaker is flapping at CLOSED <-> OPEN" (a real outage) from
    "HALF_OPEN -> CLOSED" (the dependency came back).
    """
    CIRCUIT_BREAKER_STATE.labels(name=event.name).set(_STATE_VALUE[event.to_state])

    # ``from`` is a Python keyword, so the label cannot be passed as a
    # literal keyword argument; positional labels are looked up by the
    # ``labelnames`` order declared in ``monitoring.metrics``.
    CIRCUIT_BREAKER_STATE_CHANGES_TOTAL.labels(
        event.name, event.from_state.value, event.to_state.value
    ).inc()

    if event.to_state is CircuitState.OPEN:
        CIRCUIT_BREAKER_TRIPS_TOTAL.labels(name=event.name, reason=event.reason).inc()
