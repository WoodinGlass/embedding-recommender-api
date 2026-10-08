"""Unit tests for signal providers (ADR-0016).

The synthetic and frozen providers are pure; the PostgreSQL
provider is exercised with a fake connection that mimics the two
methods it uses (``cursor`` and ``transaction``) and records the
calls, so the tests run without a database.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from collections.abc import Iterator
from typing import Any

import pytest

from recsys.retrieval.providers import (
    MAX_BATCH_SIZE,
    FrozenPopularityProvider,
    FrozenRecencyProvider,
    PgRecencyProvider,
    ProviderError,
    SyntheticPopularityProvider,
)


# ---------------------------------------------------------------- #
# synthetic + frozen
# ---------------------------------------------------------------- #
def test_synthetic_is_deterministic_across_calls() -> None:
    p = SyntheticPopularityProvider()
    assert p.get(["a", "b"]) == p.get(["a", "b"])


def test_synthetic_range() -> None:
    p = SyntheticPopularityProvider()
    for iid, value in p.get([f"i_{i}" for i in range(200)]).items():
        assert 0.0 <= value < 1000.0, iid


def test_synthetic_batch_limit() -> None:
    p = SyntheticPopularityProvider(batch_size=3)
    with pytest.raises(ProviderError, match="batch too large"):
        p.get(["a", "b", "c", "d"])


def test_frozen_popularity_uses_default() -> None:
    p = FrozenPopularityProvider({"a": 10.0}, default=0.0)
    assert p.get(["a", "b"]) == {"a": 10.0, "b": 0.0}


def test_frozen_recency_rejects_negative() -> None:
    with pytest.raises(ValueError, match="age_days"):
        FrozenRecencyProvider({"a": -1.0})


# ---------------------------------------------------------------- #
# pg recency — fake connection
# ---------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self, rows: list[tuple[str, dt.datetime]], exc: BaseException | None) -> None:
        self._rows = rows
        self._exc = exc
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        if self._exc is not None:
            raise self._exc

    def fetchall(self) -> list[tuple[str, dt.datetime]]:
        return self._rows

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeConnection:
    """Mimics psycopg's ``cursor()`` and ``transaction()``.

    ``transaction()`` is a context manager that, like the real one,
    commits on a clean exit and rolls back on exception. It records
    the outcome so a test can assert the rollback happened.
    """

    def __init__(
        self,
        rows: list[tuple[str, dt.datetime]] | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self._rows = rows or []
        self._exc = exc
        self.transactions = 0
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows, self._exc)

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        self.transactions += 1
        try:
            yield
        except BaseException:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1


def _now() -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def test_pg_recency_empty_batch_returns_empty() -> None:
    conn = _FakeConnection()
    p = PgRecencyProvider(conn, now=_now)
    assert p.get([]) == {}
    assert conn.transactions == 0


def test_pg_recency_batch_limit() -> None:
    conn = _FakeConnection()
    p = PgRecencyProvider(conn, batch_size=2, now=_now)
    with pytest.raises(ProviderError, match="batch too large"):
        p.get(["a", "b", "c"])


def test_pg_recency_returns_age_in_days() -> None:
    created = dt.datetime(2025, 12, 31, tzinfo=dt.UTC)  # 1 day before _now
    conn = _FakeConnection(rows=[("i_1", created)])
    p = PgRecencyProvider(conn, now=_now)
    ages = p.get(["i_1"])
    assert ages == {"i_1": pytest.approx(1.0)}


def test_pg_recency_uses_explicit_transaction() -> None:
    """The query runs inside ``connection.transaction()`` so a
    failure rolls back and leaves the connection usable for the
    next statement (the bug the CI run caught)."""
    created = dt.datetime(2025, 12, 31, tzinfo=dt.UTC)
    conn = _FakeConnection(rows=[("i_1", created)])
    p = PgRecencyProvider(conn, now=_now)
    p.get(["i_1"])
    assert conn.transactions == 1
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_pg_recency_failure_rolls_back_and_wraps() -> None:
    """A failing query is wrapped in ``ProviderError`` and the
    transaction is rolled back, so the shared connection is not
    left in the aborted state."""
    conn = _FakeConnection(exc=RuntimeError("boom"))
    p = PgRecencyProvider(conn, now=_now)
    with pytest.raises(ProviderError, match="RuntimeError"):
        p.get(["i_1"])
    assert conn.transactions == 1
    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_pg_recency_skips_null_created_at() -> None:
    rows = [("i_1", None), ("i_2", dt.datetime(2025, 12, 31, tzinfo=dt.UTC))]
    conn = _FakeConnection(rows=rows)  # type: ignore[arg-type]
    p = PgRecencyProvider(conn, now=_now)
    ages = p.get(["i_1", "i_2"])
    assert "i_1" not in ages
    assert "i_2" in ages


def test_pg_recency_clamps_future_created_at() -> None:
    """A ``created_at`` in the future would make ``age_days``
    negative; the provider clamps to 0 rather than raising."""
    future = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)
    conn = _FakeConnection(rows=[("i_1", future)])
    p = PgRecencyProvider(conn, now=_now)
    assert p.get(["i_1"]) == {"i_1": 0.0}


def test_pg_recency_timeout_must_be_positive() -> None:
    conn = _FakeConnection()
    with pytest.raises(ValueError, match="timeout_seconds"):
        PgRecencyProvider(conn, timeout_seconds=0.0)


def test_max_batch_size_is_a_positive_constant() -> None:
    assert isinstance(MAX_BATCH_SIZE, int)
    assert MAX_BATCH_SIZE > 0
