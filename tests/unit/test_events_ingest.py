"""Unit tests for batch event ingestion (ADR-0018).

A fake connection records the SQL and parameters so the tests run
without a database and can assert the PII property (the raw user id
never reaches the driver). The integration test
(``tests/integration/test_event_ingest.py``) exercises the real
`ON CONFLICT DO NOTHING` behavior.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from recsys.api.schemas.events import EventEnvelope, ExperimentRef
from recsys.events.ingest import (
    IngestResult,
    IngestValidationError,
    ingest_events,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------- #
# fakes
# ---------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self, *, rowcount: int | None = None) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.batch: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.rowcount = rowcount if rowcount is not None else -1

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def executemany(self, sql: str, params_iter: Any) -> None:
        rows = list(params_iter)
        self.batch.append((sql, rows))
        if self.rowcount < 0:
            # Default: pretend every row was inserted.
            self.rowcount = len(rows)

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeConnection:
    def __init__(self, *, rowcount: int | None = None) -> None:
        self._rowcount = rowcount
        self.transactions = 0
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(rowcount=self._rowcount)

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


# ---------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------- #
def _envelope(
    *,
    event_id: str | None = "ev-12345678",
    event_ts: datetime | None = None,
    user_id: str = "u_1",
    event_type: str = "impression",
    item_id: str = "i_1",
    request_id: str | None = None,
    experiment: ExperimentRef | None = None,
    position: int | None = 1,
    value: float | None = None,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_ts=event_ts or NOW,
        event_type=event_type,  # type: ignore[arg-type]
        user_id=user_id,
        item_id=item_id,
        request_id=request_id,
        experiment=experiment,
        position=position,
        value=value,
    )


def _ingest(
    *,
    events: list[EventEnvelope],
    request_id: str = "req-1",
    now: datetime = NOW,
    max_age_seconds: int = 604_800,
    max_future_seconds: int = 300,
    user_id_salt: str = "salt",
    user_id_salt_version: int = 1,
    connection: _FakeConnection | None = None,
) -> tuple[IngestResult, _FakeConnection]:
    conn = connection or _FakeConnection()
    result = ingest_events(
        conn,
        events=events,
        request_id=request_id,
        now=now,
        max_age_seconds=max_age_seconds,
        max_future_seconds=max_future_seconds,
        user_id_salt=user_id_salt,
        user_id_salt_version=user_id_salt_version,
    )
    return result, conn


# ---------------------------------------------------------------- #
# happy path
# ---------------------------------------------------------------- #
def test_single_event_accepted() -> None:
    result, conn = _ingest(events=[_envelope()])
    assert result.accepted == 1
    assert result.duplicates == 0
    assert result.rejected == 0
    assert result.generated_event_ids is None
    assert conn.transactions == 1
    assert conn.commits == 1


def test_batch_accepted() -> None:
    events = [_envelope(event_id=f"ev-1234567{i}") for i in range(5)]
    result, _ = _ingest(events=events)
    assert result.accepted == 5


# ---------------------------------------------------------------- #
# generated ids
# ---------------------------------------------------------------- #
def test_generated_id_when_client_omits() -> None:
    result, _ = _ingest(events=[_envelope(event_id=None)])
    assert result.generated_event_ids is not None
    assert len(result.generated_event_ids) == 1
    assert len(result.generated_event_ids[0]) == 64


def test_generated_ids_parallel_to_omitted_events() -> None:
    events = [
        _envelope(event_id="ev-12345678"),
        _envelope(event_id=None),
        _envelope(event_id="ev-99999999"),
        _envelope(event_id=None),
    ]
    result, _ = _ingest(events=events)
    assert result.generated_event_ids is not None
    assert len(result.generated_event_ids) == 2


def test_generated_id_is_stable_for_same_request_and_index() -> None:
    """A retry of the same request with the same request_id produces
    the same generated ids, so the retry's insert is idempotent."""
    a, _ = _ingest(events=[_envelope(event_id=None)], request_id="req-X")
    b, _ = _ingest(events=[_envelope(event_id=None)], request_id="req-X")
    assert a.generated_event_ids == b.generated_event_ids


def test_generated_id_differs_across_request_ids() -> None:
    a, _ = _ingest(events=[_envelope(event_id=None)], request_id="req-A")
    b, _ = _ingest(events=[_envelope(event_id=None)], request_id="req-B")
    assert a.generated_event_ids != b.generated_event_ids


# ---------------------------------------------------------------- #
# skew window
# ---------------------------------------------------------------- #
def test_event_too_old_rejected() -> None:
    old = NOW - timedelta(days=10)
    with pytest.raises(IngestValidationError) as ei:
        _ingest(events=[_envelope(event_ts=old)])
    assert ei.value.field == "event_ts"
    assert ei.value.index == 0


def test_event_too_future_rejected() -> None:
    future = NOW + timedelta(minutes=10)
    with pytest.raises(IngestValidationError):
        _ingest(events=[_envelope(event_ts=future)])


def test_naive_timestamp_rejected() -> None:
    naive = datetime(2026, 1, 1)
    with pytest.raises(IngestValidationError, match="timezone-aware"):
        _ingest(events=[_envelope(event_ts=naive)])


def test_boundary_oldest_accepted() -> None:
    edge = NOW - timedelta(days=7)
    result, _ = _ingest(events=[_envelope(event_ts=edge)])
    assert result.accepted == 1


def test_boundary_newest_accepted() -> None:
    edge = NOW + timedelta(seconds=300)
    result, _ = _ingest(events=[_envelope(event_ts=edge)])
    assert result.accepted == 1


def test_second_event_invalid_reports_its_index() -> None:
    old = NOW - timedelta(days=10)
    events = [_envelope(event_id="ev-12345678"), _envelope(event_ts=old, event_id="ev-12345679")]
    with pytest.raises(IngestValidationError) as ei:
        _ingest(events=events)
    assert ei.value.index == 1


# ---------------------------------------------------------------- #
# duplicates (fake with rowcount=0)
# ---------------------------------------------------------------- #
def test_duplicate_counted_when_rowcount_lower() -> None:
    conn = _FakeConnection(rowcount=1)  # 3 sent, only 1 inserted
    result, _ = _ingest(
        events=[_envelope(event_id=f"ev-1234567{i}") for i in range(3)],
        connection=conn,
    )
    assert result.accepted == 1
    assert result.duplicates == 2


# ---------------------------------------------------------------- #
# PII redaction
# ---------------------------------------------------------------- #
def test_raw_user_id_never_reaches_driver() -> None:
    """The raw `user_id` value must not appear in the SQL string or
    in any parameter sent to the database (ADR-0018 § PII redaction
    is structural)."""
    sentinel = "SENTINEL_USER_ID_abc123"
    _ingest(events=[_envelope(user_id=sentinel)])
    # We cannot reach into the fake's recorded state after the call
    # because the ingest uses a fresh cursor. Run again with a
    # capture cursor.
    captured_sql: list[str] = []
    captured_params: list[Any] = []

    class CaptureCursor(_FakeCursor):
        def execute(self, sql: str, params: Any = None) -> None:
            captured_sql.append(sql)
            captured_params.append(params)
            super().execute(sql, params)

        def executemany(self, sql: str, params_iter: Any) -> None:
            captured_sql.append(sql)
            rows = list(params_iter)
            captured_params.append(rows)
            super().executemany(sql, rows)

    class CaptureConn(_FakeConnection):
        def cursor(self) -> CaptureCursor:
            return CaptureCursor(rowcount=1)

    _ingest(events=[_envelope(user_id=sentinel)], connection=CaptureConn())
    flat = " ".join(captured_sql) + " " + repr(captured_params)
    assert sentinel not in flat


# ---------------------------------------------------------------- #
# guards
# ---------------------------------------------------------------- #
def test_empty_events_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _ingest(events=[])


def test_empty_request_id_raises() -> None:
    with pytest.raises(ValueError, match="request_id"):
        _ingest(events=[_envelope()], request_id="")


def test_negative_max_age_raises() -> None:
    with pytest.raises(ValueError, match="max_age_seconds"):
        _ingest(events=[_envelope()], max_age_seconds=-1)
