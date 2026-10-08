"""Signal providers for the re-ranker (ADR-0016).

Two signals the re-ranker blends are not on the embedding: popularity
(how many interactions the item has accumulated) and recency (how
recently the item was introduced). Both come from outside the
retrieval layer and both are fetched by a **provider**.

Providers are batch-only. A provider that fetches one id per call
would pay a round trip per candidate; the candidate window is
``4 × k`` items, so a request that re-ranks 100 candidates would
make 400 round trips. ``get(item_ids)`` returns a mapping, and the
caller does one call per signal per request.

Providers are synchronous. ``PgvectorBackend`` is synchronous (ADR-0012)
and the retrieval path already pays the round trip on one connection;
a provider that returned a coroutine would force the caller to mix
sync and async for no benefit.

Providers raise. They do not return a sentinel for "unavailable",
they do not know about neutral values, and they do not log. The
re-ranker's failure contract (ADR-0016 § Failure behavior) is the
caller's job: a provider that raises or times out produces the
neutral value for every candidate and increments
``recsys_rerank_signal_missing_total{signal}``. Keeping the neutral
policy in one place (the re-ranker) is what makes the "one code path
for no information" rule hold.

Determinism. A provider that returns a different value for the same
item on two calls within one process is a bug (ADR-0016 §
Determinism). The synthetic provider hashes the item id; a provider
that reads a mutable counter or a clock would break the M2
evaluation gate and the M5 A/B assignment.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol, runtime_checkable

#: Maximum number of ids a single ``get`` call accepts. The candidate
#: window is ``candidate_multiplier × k`` (config); with the default
#: multiplier of 4, a request for 100 recommendations fetches 400
#: candidates, which is well under this bound. The limit exists so a
#: misconfigured multiplier cannot turn one provider call into a
#: query plan the database refuses.
MAX_BATCH_SIZE: Final[int] = 1000

#: The synthetic popularity range. The exact numbers do not matter:
#: the re-ranker normalizes popularity by rank within the window, so
#: only the ordering is read. A range of 1000 gives a collision
#: probability of roughly one in a thousand for two random items,
#: which is fine for a placeholder.
_SYNTHETIC_POPULARITY_RANGE: Final[int] = 1000

#: Number of hex characters of the SHA-256 digest used to derive a
#: synthetic popularity. Eight hex chars = 32 bits, which covers the
#: 1000-value range with no observable bias.
_SYNTHETIC_HASH_CHARS: Final[int] = 8


class ProviderError(RuntimeError):
    """A provider could not produce a value.

    Raised for a timeout, a connection error, a malformed result, or
    an oversized batch. The re-ranker treats every subclass the same
    way: signal neutral, request proceeds, metric incremented.
    """


# ------------------------------------------------------------------ #
# popularity
# ------------------------------------------------------------------ #
@runtime_checkable
class PopularityProvider(Protocol):
    """Return a popularity score for each requested item.

    The value's scale is not specified; the re-ranker normalizes by
    rank within the candidate window, so only the ordering is read.
    A missing id produces no entry in the mapping; the caller treats
    a missing entry the same as a neutral value for that item.
    """

    name: str

    def get(self, item_ids: Sequence[str]) -> Mapping[str, float]:
        """Return ``{item_id: popularity}`` for the requested ids."""
        ...


def _synthetic_popularity(item_id: str) -> float:
    """Deterministic pseudo-popularity in ``[0, 1000)`` for an item.

    Uses SHA-256, not Python's built-in ``hash``: ``hash`` is
    randomized per process (``PYTHONHASHSEED``), which would make the
    value non-deterministic across processes and break the M2
    evaluation gate and the M5 A/B assignment (ADR-0016 §
    Determinism). SHA-256 is stable across processes, Python
    versions, and platforms.
    """
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()
    head = digest[:_SYNTHETIC_HASH_CHARS]
    return float(int(head, 16) % _SYNTHETIC_POPULARITY_RANGE)


class SyntheticPopularityProvider:
    """Placeholder popularity provider for M3.

    The value is a deterministic hash of the item id. It has no
    relationship to how many interactions an item actually has —
    there is no event log until M5 — and the ADR records it as a
    placeholder. Do not draw a business conclusion from this
    provider's output; it exists so that the re-ranker's blend can
    be exercised end to end before M5 lands a real signal.

    M5 replaces this with a provider backed by the event log; the
    interface is unchanged.
    """

    name: str = "synthetic"

    def __init__(self, *, batch_size: int = MAX_BATCH_SIZE) -> None:
        if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}], got {batch_size!r}")
        self._batch_size = batch_size

    def get(self, item_ids: Sequence[str]) -> Mapping[str, float]:
        self._check_batch(item_ids)
        return {item_id: _synthetic_popularity(item_id) for item_id in item_ids}

    def _check_batch(self, item_ids: Sequence[str]) -> None:
        if len(item_ids) > self._batch_size:
            raise ProviderError(f"batch too large: {len(item_ids)} > {self._batch_size}")


class FrozenPopularityProvider:
    """A test provider that returns a fixed mapping.

    Every requested id gets a value: the mapping's value if present,
    otherwise ``default``. This makes the provider's output
    independent of the caller's iteration order, which is what a
    test needs when it asserts a specific re-ranking.
    """

    name: str = "frozen"

    def __init__(
        self,
        scores: Mapping[str, float],
        *,
        default: float = 0.0,
        batch_size: int = MAX_BATCH_SIZE,
    ) -> None:
        if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}], got {batch_size!r}")
        self._scores = dict(scores)
        self._default = default
        self._batch_size = batch_size

    def get(self, item_ids: Sequence[str]) -> Mapping[str, float]:
        if len(item_ids) > self._batch_size:
            raise ProviderError(f"batch too large: {len(item_ids)} > {self._batch_size}")
        return {item_id: self._scores.get(item_id, self._default) for item_id in item_ids}


# ------------------------------------------------------------------ #
# recency
# ------------------------------------------------------------------ #
@runtime_checkable
class RecencyProvider(Protocol):
    """Return an age in days for each requested item.

    Age is measured from ``item.created_at`` (ADR-0016 § Step 1). A
    missing id produces no entry; the caller treats a missing entry
    as a neutral value for that item.
    """

    name: str

    def get(self, item_ids: Sequence[str]) -> Mapping[str, float]:
        """Return ``{item_id: age_days}`` for the requested ids."""
        ...


class FrozenRecencyProvider:
    """A test provider that returns a fixed mapping."""

    name: str = "frozen"

    def __init__(
        self,
        ages_days: Mapping[str, float],
        *,
        default: float = 0.0,
        batch_size: int = MAX_BATCH_SIZE,
    ) -> None:
        if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}], got {batch_size!r}")
        for item_id, age in ages_days.items():
            if age < 0:
                raise ValueError(f"age_days for {item_id!r} must be >= 0, got {age!r}")
        self._ages = dict(ages_days)
        self._default = default
        self._batch_size = batch_size

    def get(self, item_ids: Sequence[str]) -> Mapping[str, float]:
        if len(item_ids) > self._batch_size:
            raise ProviderError(f"batch too large: {len(item_ids)} > {self._batch_size}")
        return {item_id: self._ages.get(item_id, self._default) for item_id in item_ids}


class PgRecencyProvider:
    """Recency provider backed by the ``item`` table.

    One query per batch: ``SELECT item_id, created_at FROM item WHERE
    item_id = ANY(%s)``. ``item_id`` is the primary key, so the plan
    is an index scan, not a sequential scan.

    The connection is passed in, not created here. The caller (the
    serving layer, or a test with a fake connection) owns the
    lifecycle, matching the pattern ``PgvectorBackend`` uses
    (ADR-0012). The provider is a pure strategy over a database.

    The per-call timeout is set with ``SET LOCAL statement_timeout``
    so it applies to this transaction only. ``SET LOCAL`` is
    PostgreSQL-specific and is the correct tool: a client-side
    timeout would leave the query running server-side, and a
    session-level ``SET`` would outlive the call.
    """

    name: str = "postgres"

    def __init__(
        self,
        connection: Any,
        *,
        timeout_seconds: float = 1.0,
        batch_size: int = MAX_BATCH_SIZE,
        now: Any = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")
        if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}], got {batch_size!r}")
        self._connection = connection
        self._timeout_ms = max(1, int(timeout_seconds * 1000))
        self._batch_size = batch_size
        self._now = now or (lambda: _dt.datetime.now(_dt.UTC))

    def get(self, item_ids: Sequence[str]) -> Mapping[str, float]:
        if not item_ids:
            return {}
        if len(item_ids) > self._batch_size:
            raise ProviderError(f"batch too large: {len(item_ids)} > {self._batch_size}")

        # The query runs inside an explicit transaction. Two reasons:
        #
        # 1. ``SET LOCAL`` is scoped to the current transaction; outside
        #    one, psycopg begins an implicit transaction on the first
        #    statement and leaves it open. If the query then fails, the
        #    connection is left in the "aborted" state and the next
        #    statement on the *same connection* fails with
        #    ``InFailedSqlTransaction``. The provider shares its
        #    connection with ``PgvectorBackend`` in the evaluation script,
        #    so a failed recency query would take down the retrieval
        #    query that follows.
        #
        # 2. ``with connection.transaction()`` commits on a clean exit
        #    and rolls back on an exception. The rollback is what returns
        #    the connection to a usable state; it is the fix, not a
        #    side effect.
        try:
            with self._connection.transaction(), self._connection.cursor() as cur:
                cur.execute(
                    "SET LOCAL statement_timeout = %s",
                    (self._timeout_ms,),
                )
                cur.execute(
                    "SELECT item_id, created_at FROM item WHERE item_id = ANY(%s)",
                    (list(item_ids),),
                )
                rows = cur.fetchall()
        except Exception as exc:
            raise ProviderError(f"recency query failed: {type(exc).__name__}") from exc

        now = self._now()
        ages: dict[str, float] = {}
        for item_id, created_at in rows:
            if created_at is None:
                continue
            if created_at.tzinfo is None:
                # Defensive: the column is TIMESTAMPTZ and psycopg
                # returns a tz-aware datetime. A naive one would mean
                # the connection was configured to drop tz; treat the
                # value as UTC rather than let subtraction fail.
                created_at = created_at.replace(tzinfo=_dt.UTC)
            delta = now - created_at
            ages[str(item_id)] = max(0.0, delta.total_seconds() / 86_400.0)
        return ages


__all__ = [
    "MAX_BATCH_SIZE",
    "FrozenPopularityProvider",
    "FrozenRecencyProvider",
    "PgRecencyProvider",
    "PopularityProvider",
    "ProviderError",
    "RecencyProvider",
    "SyntheticPopularityProvider",
]
