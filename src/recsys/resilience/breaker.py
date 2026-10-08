"""Circuit breaker (ADR-0015).

One breaker per external dependency. The cache and the limiter share
the Redis breaker; PostgreSQL has its own. The breaker's job is to
decide whether to try a call, not what to do when the call fails.

Design choices:

- **Per-instance state.** ``failure_count``, ``state``, and
  ``opened_at`` live in the process. Sharing them via Redis would
  mean consulting the dependency to decide whether the dependency
  is down.
- **Failure classification is explicit.** Only exceptions that mean
  the dependency did not answer count as failures: builtin
  ``ConnectionError`` / ``TimeoutError`` / ``asyncio.TimeoutError``,
  and any exception whose class name is in the allowlist below
  (``redis.ConnectionError``, ``redis.TimeoutError``,
  ``psycopg.OperationalError``, ...). ``ValueError``, ``TypeError``,
  ``redis.ResponseError``, and any error raised by a response that
  arrived intact are **not** failures; they are re-raised to the
  caller without touching the breaker. Without this rule, a logic
  bug in one path would open the breaker for the other.
- **Single probe in HALF_OPEN.** The first call after the timer
  allows one probe. A concurrent second call while the probe is in
  flight is rejected with ``CircuitOpenError``. Letting N probes
  through would send a load spike at a dependency that just came
  back.
- **Exponential backoff.** The first OPEN lasts ``open_seconds``;
  each subsequent trip without an intervening success doubles it up
  to ``open_max_seconds``. A success resets the backoff.
- **Clock is injected.** Tests control time via a fake clock; the
  production default is ``time.monotonic`` (immune to wall-clock
  jumps from NTP or a manual ``date``).
- **No import of ``monitoring``.** The breaker reports state
  transitions through an optional callback. Wiring that callback to
  a Prometheus metric happens in the composition root, not here;
  the breaker does not know it is being measured.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, TypeVar

T = TypeVar("T")

#: Exception class names that count as a failure of the dependency.
#: A class name is enough because the alternative — importing
#: ``redis`` and ``psycopg`` here — would force every caller of the
#: breaker to install both. The names are chosen so a genuine bug in
#: a caller (``ValueError``, ``TypeError``, ``KeyError``) never
#: matches. ``redis.ResponseError`` (a valid but unexpected reply)
#: and ``psycopg.ProgrammingError`` (a SQL error) are deliberately
#: absent: they mean the dependency answered, so it is up.
_FAILURE_CLASS_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "TimeoutError",
        "TimeoutExpired",
        "OperationalError",  # psycopg (connect / server gone)
        "InterfaceError",  # psycopg (connection unusable)
        "BusyLoadingError",  # redis (still loading dataset)
        "ConnectionPoolExhausted",  # pool timeout, sometimes raised by wrappers
    }
)


class CircuitState(StrEnum):
    """The three states the breaker moves between."""

    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is refused because the breaker is open.

    The caller treats this as "the dependency is not available right
    now" and chooses the fallback appropriate to its own path: the
    cache treats it as a miss, the limiter uses its per-instance
    bucket, a caller with no fallback lets the error surface.
    """

    def __init__(self, name: str, *, reason: str = "open") -> None:
        super().__init__(f"circuit breaker {name!r} is open ({reason})")
        self.name = name
        self.reason = reason


@dataclass(frozen=True)
class TransitionEvent:
    """A single state transition, passed to ``on_transition``."""

    name: str
    from_state: CircuitState
    to_state: CircuitState
    reason: str


def _is_failure(exc: BaseException) -> bool:
    """Return True when ``exc`` means the dependency did not answer.

    Connection and timeout errors qualify. A response that arrived
    intact — even a bad one — does not. See the module docstring and
    ADR-0015 for the reasoning.
    """
    # ``asyncio.TimeoutError`` is an alias of the builtin in 3.11+;
    # the tuple covers both spellings on any interpreter.
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
        return True
    return type(exc).__name__ in _FAILURE_CLASS_NAMES


class CircuitBreaker:
    """A single breaker guarding a single dependency.

    The instance is not thread-safe across OS threads; it is safe
    across coroutines on one event loop (the ``_lock`` serializes
    the state transitions).
    """

    def __init__(
        self,
        *,
        name: str,
        failure_threshold: int,
        open_seconds: float,
        open_max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        on_transition: Callable[[TransitionEvent], None] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if open_seconds <= 0:
            raise ValueError(f"open_seconds must be > 0, got {open_seconds}")
        if open_max_seconds < open_seconds:
            raise ValueError(
                f"open_max_seconds ({open_max_seconds}) must be >= open_seconds ({open_seconds})"
            )

        self._name = name
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._open_max_seconds = open_max_seconds
        self._clock = clock
        self._on_transition = on_transition

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float = 0.0
        self._current_open_seconds = open_seconds
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ #
    # read-only views
    # ------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self._name

    def state(self) -> CircuitState:
        """Return the current state.

        OPEN is reported as OPEN even if its timer has elapsed; the
        transition to HALF_OPEN happens inside :meth:`call` because
        it is the first call that acts on the elapsed timer.
        """
        return self._state

    def is_open(self) -> bool:
        """Return True when the breaker is refusing calls.

        A caller that only wants a fast yes/no (the cache path does)
        checks this before building a coroutine; a caller that wants
        to participate in the HALF_OPEN probe uses :meth:`call`.
        """
        return self._state is CircuitState.OPEN

    # ------------------------------------------------------------ #
    # call path
    # ------------------------------------------------------------ #
    async def call(self, coro: Awaitable[T]) -> T:
        """Run ``coro`` under the breaker.

        Raises :class:`CircuitOpenError` if the breaker refuses the
        call (OPEN, or HALF_OPEN with a probe already in flight).
        Re-raises the original exception if ``coro`` raises a
        non-failure (a ``ValueError`` from a caller bug); the
        breaker state is unchanged in that case.
        """
        async with self._lock:
            self._admit_or_raise_locked()

        try:
            result = await coro
        except BaseException as exc:
            async with self._lock:
                if not _is_failure(exc):
                    # Not a dependency failure; leave state alone.
                    raise
                self._on_failure_locked()
            raise
        else:
            async with self._lock:
                self._on_success_locked()
            return result

    # ------------------------------------------------------------ #
    # transitions (all called with self._lock held)
    # ------------------------------------------------------------ #
    def _admit_or_raise_locked(self) -> None:
        now = self._clock()
        if self._state is CircuitState.OPEN:
            elapsed = now - self._opened_at
            if elapsed < self._current_open_seconds:
                raise CircuitOpenError(self._name, reason="open")
            # Timer elapsed: this call becomes the single probe.
            self._transition_locked(CircuitState.HALF_OPEN, reason="probe")
            return
        if self._state is CircuitState.HALF_OPEN:
            # A probe is already running; do not send a second one.
            raise CircuitOpenError(self._name, reason="probe_in_flight")

    def _on_success_locked(self) -> None:
        if self._state is CircuitState.CLOSED:
            # A success in CLOSED resets the consecutive counter: the
            # threshold counts failures *in a row*.
            self._failure_count = 0
            return
        # Success while HALF_OPEN: close and reset the backoff.
        self._failure_count = 0
        self._current_open_seconds = self._open_seconds
        self._transition_locked(CircuitState.CLOSED, reason="probe_succeeded")

    def _on_failure_locked(self) -> None:
        self._failure_count += 1
        if self._state is CircuitState.CLOSED:
            if self._failure_count >= self._failure_threshold:
                self._open_locked(reason="threshold")
            return
        if self._state is CircuitState.HALF_OPEN:
            # A failed probe re-opens and lengthens the backoff.
            self._open_locked(reason="probe_failed")

    def _open_locked(self, *, reason: str) -> None:
        self._opened_at = self._clock()
        self._transition_locked(CircuitState.OPEN, reason=reason)
        # Extend the backoff only on consecutive trips; a successful
        # probe resets _current_open_seconds in _on_success_locked.
        if reason == "probe_failed":
            self._current_open_seconds = min(self._current_open_seconds * 2, self._open_max_seconds)
        elif self._failure_count >= self._failure_threshold * 2:
            # A second threshold-crossing without a success means the
            # breaker is flapping; start the backoff.
            self._current_open_seconds = min(self._current_open_seconds * 2, self._open_max_seconds)

    def _transition_locked(self, to_state: CircuitState, *, reason: str) -> None:
        from_state = self._state
        if from_state is to_state:
            return
        self._state = to_state
        if self._on_transition is not None:
            # A callback that raises must not corrupt breaker state.
            # The callback is observability; the state is behavior.
            with contextlib.suppress(Exception):
                self._on_transition(
                    TransitionEvent(
                        name=self._name,
                        from_state=from_state,
                        to_state=to_state,
                        reason=reason,
                    )
                )
