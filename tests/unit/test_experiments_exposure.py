"""Unit tests for the exposure writer (ADR-0017).

Uses a fake connection that mimics ``cursor()`` and ``transaction()``
and records the SQL and parameters; no database. The integration
test (``tests/integration/test_exposure_writer.py``) exercises the
real path against PostgreSQL in CI.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest

from recsys.events import UserIdHash
from recsys.experiments.assignment import Assignment
from recsys.experiments.exposure import (
    ExposureResult,
    ExposureWriteError,
    write_exposure,
)


# ---------------------------------------------------------------- #
# fakes
# ---------------------------------------------------------------- #
class _FakeCursor:
    def __init__(
        self,
        *,
        rowcount: int = 1,
        raise_on: str | None = None,
        fail_with: BaseException | None = None,
    ) -> None:
        self._rowcount = rowcount
        self._raise_on = raise_on
        self._fail_with = fail_with
        self.executed: list[tuple[str, Any]] = []
        self.rowcount = -1

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        if self._raise_on and self._raise_on in sql:
            assert self._fail_with is not None
            raise self._fail_with
        # After a successful INSERT, set rowcount to what the fake
        # was configured with.
        if "INSERT INTO experiment_exposure" in sql:
            self.rowcount = self._rowcount

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeConnection:
    def __init__(
        self,
        *,
        rowcount: int = 1,
        raise_on: str | None = None,
        fail_with: BaseException | None = None,
    ) -> None:
        self._rowcount = rowcount
        self._raise_on = raise_on
        self._fail_with = fail_with
        self.transactions = 0
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(
            rowcount=self._rowcount,
            raise_on=self._raise_on,
            fail_with=self._fail_with,
        )

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
def _assignment(
    *,
    name: str = "exp",
    variant: str = "control",
    bucket: int = 1234,
    effective_salt: str = "prod:salt-v1",
    paused: bool = False,
    fallback_reason: str | None = None,
    log_exposure: bool = True,
) -> Assignment:
    return Assignment(
        experiment_name=name,
        variant=variant,
        bucket=bucket,
        effective_salt=effective_salt,
        paused=paused,
        fallback_reason=fallback_reason,
        log_exposure=log_exposure,
    )


def _hash() -> UserIdHash:
    return UserIdHash(hex="a" * 64, version=1)


# ---------------------------------------------------------------- #
# happy path
# ---------------------------------------------------------------- #
def test_insert_success_returns_inserted_true() -> None:
    conn = _FakeConnection(rowcount=1)
    result = write_exposure(
        connection=conn,
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    assert isinstance(result, ExposureResult)
    assert result.inserted is True
    assert result.skipped_reason is None
    # 64 lowercase hex
    assert len(result.event_id) == 64
    assert all(c in "0123456789abcdef" for c in result.event_id)
    # One transaction, committed, no rollback
    assert conn.transactions == 1
    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_event_id_is_deterministic() -> None:
    a = write_exposure(
        connection=_FakeConnection(),
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    b = write_exposure(
        connection=_FakeConnection(),
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    assert a.event_id == b.event_id


def test_event_id_changes_with_request_id() -> None:
    a = write_exposure(
        connection=_FakeConnection(),
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    b = write_exposure(
        connection=_FakeConnection(),
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id="req-2",
    )
    assert a.event_id != b.event_id


def test_event_id_changes_with_experiment() -> None:
    a = write_exposure(
        connection=_FakeConnection(),
        assignment=_assignment(name="exp_a"),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    b = write_exposure(
        connection=_FakeConnection(),
        assignment=_assignment(name="exp_b"),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    assert a.event_id != b.event_id


# ---------------------------------------------------------------- #
# skip when log_exposure is False
# ---------------------------------------------------------------- #
def test_skips_when_log_exposure_false() -> None:
    conn = _FakeConnection()
    result = write_exposure(
        connection=conn,
        assignment=_assignment(log_exposure=False, fallback_reason="stopped"),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    assert result.inserted is False
    assert result.skipped_reason == "log_exposure_false"
    # No database call.
    assert conn.transactions == 0


# ---------------------------------------------------------------- #
# duplicate
# ---------------------------------------------------------------- #
def test_duplicate_returns_inserted_false() -> None:
    """`rowcount == 0` from `ON CONFLICT DO NOTHING` means a row
    with this `event_id` already exists: a retried request. The
    result is a clean "not inserted", not an error."""
    conn = _FakeConnection(rowcount=0)
    result = write_exposure(
        connection=conn,
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    assert result.inserted is False
    assert result.skipped_reason == "duplicate"
    assert conn.transactions == 1
    assert conn.commits == 1


# ---------------------------------------------------------------- #
# failures
# ---------------------------------------------------------------- #
def test_insert_failure_wraps_and_rolls_back() -> None:
    conn = _FakeConnection(
        raise_on="INSERT INTO experiment_exposure",
        fail_with=RuntimeError("boom"),
    )
    with pytest.raises(ExposureWriteError, match="RuntimeError"):
        write_exposure(
            connection=conn,
            assignment=_assignment(),
            user_id_hash=_hash(),
            request_id="req-1",
        )
    assert conn.transactions == 1
    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_set_local_failure_also_rolls_back() -> None:
    conn = _FakeConnection(
        raise_on="SET LOCAL",
        fail_with=RuntimeError("timeout unsupported"),
    )
    with pytest.raises(ExposureWriteError):
        write_exposure(
            connection=conn,
            assignment=_assignment(),
            user_id_hash=_hash(),
            request_id="req-1",
        )
    assert conn.rollbacks == 1


# ---------------------------------------------------------------- #
# guards
# ---------------------------------------------------------------- #
def test_zero_timeout_raises() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        write_exposure(
            connection=_FakeConnection(),
            assignment=_assignment(),
            user_id_hash=_hash(),
            request_id="req-1",
            timeout_seconds=0.0,
        )


def test_empty_request_id_raises() -> None:
    with pytest.raises(ValueError, match="request_id"):
        write_exposure(
            connection=_FakeConnection(),
            assignment=_assignment(),
            user_id_hash=_hash(),
            request_id="",
        )


def test_empty_experiment_name_raises() -> None:
    with pytest.raises(ValueError, match="experiment"):
        write_exposure(
            connection=_FakeConnection(),
            assignment=_assignment(name=""),
            user_id_hash=_hash(),
            request_id="req-1",
        )


# ---------------------------------------------------------------- #
# SQL shape (light check: the writer uses parameterized SQL)
# ---------------------------------------------------------------- #
def test_insert_sql_is_parameterized() -> None:
    conn = _FakeConnection()
    write_exposure(
        connection=conn,
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    # Re-run to capture executed statements via the same fake used
    # inside the writer. The writer makes a new cursor; we assert
    # by re-reading a fresh call's statements would not work, so
    # the assertion is on the writer's behavior through `cursor()`.
    # Simplest valid check: the fake's `execute` was called with a
    # parameter list, not a formatted string. Rebuild a fake and
    # inspect.
    captured: list[tuple[str, Any]] = []

    class CaptureCursor(_FakeCursor):
        def execute(self, sql: str, params: Any = None) -> None:
            captured.append((sql, params))
            super().execute(sql, params)

    class CaptureConn(_FakeConnection):
        def cursor(self) -> CaptureCursor:
            return CaptureCursor(rowcount=1)

    write_exposure(
        connection=CaptureConn(),
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id="req-1",
    )
    insert_statements = [s for s, _ in captured if "INSERT INTO experiment_exposure" in s]
    assert len(insert_statements) == 1
    assert "%s" in insert_statements[0]
    # The request_id value is passed as a parameter, not formatted in.
    assert "req-1" not in insert_statements[0]
