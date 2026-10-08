"""Resilience primitives (circuit breaker now; retry, bulkhead later)."""

from recsys.resilience.breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    TransitionEvent,
)

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "TransitionEvent",
]
