"""Integration test for batch event ingestion (ADR-0018)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from recsys.api.schemas.events import EventEnvelope
from recsys.events.ingest import IngestValidationError, ingest_events

pytestmark = [pytest.mark.integration]

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def db_url() -> str:
    url = os.environ.get("RECSYS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set RECSYS_TEST_DATABASE_URL to run postgres integration tests")
    return url


@pytest.fixture
def conn(db_url: str) -> Generator[Any, None, None]:
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(db_url) as c:
        yield c


def _envelope(*, event_id: str | None = None, event_ts: datetime | None = None) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id,
        event_ts=event_ts or NOW,
        event_type="impression",
        user_id="u_test",
        item_id="i_test",
        position=1,
    )


def _uid() -> str:
    return f"it-{uuid.uuid4().hex}"


def _ingest(
    conn: Any,
    events: list[EventEnvelope],
    *,
    request_id: str | None = None,
) -> Any:
    """Call the ingester with the integration tests' defaults.

    A narrow keyword surface (only ``request_id`` is overridable) keeps
    mypy able to check the call; a ``**overrides: Any`` signature would
    widen every keyword to ``object`` for a flexibility the tests do
    not use.
    """
    return ingest_events(
        conn,
        events=events,
        request_id=request_id or _uid(),
        now=NOW,
        max_age_seconds=604_800,
        max_future_seconds=300,
        user_id_salt="test-salt",
        user_id_salt_version=1,
    )


def _cleanup(conn: Any, ids: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM event WHERE event_id = ANY(%s)", (ids,))


def test_single_event_inserted(conn: Any) -> None:
    eid = _uid()
    try:
        result = _ingest(conn, [_envelope(event_id=eid)])
        assert result.accepted == 1
        assert result.duplicates == 0
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM event WHERE event_id = %s", (eid,))
            assert int(cur.fetchone()[0]) == 1
    finally:
        _cleanup(conn, [eid])


def test_retry_is_duplicate(conn: Any) -> None:
    eid = _uid()
    try:
        a = _ingest(conn, [_envelope(event_id=eid)])
        b = _ingest(conn, [_envelope(event_id=eid)])
        assert a.accepted == 1
        assert b.accepted == 0
        assert b.duplicates == 1
    finally:
        _cleanup(conn, [eid])


def test_batch_with_generated_ids_is_idempotent(conn: Any) -> None:
    """Two ingests of the same events with the same `request_id`
    produce the same generated ids and the second is a no-op."""
    req = _uid()
    events = [_envelope(), _envelope()]
    try:
        a = _ingest(conn, events, request_id=req)
        assert a.accepted == 2
        assert a.generated_event_ids is not None
        b = _ingest(conn, events, request_id=req)
        assert b.accepted == 0
        assert b.duplicates == 2
        assert a.generated_event_ids == b.generated_event_ids
    finally:
        with conn.cursor() as cur:
            if a.generated_event_ids:
                cur.execute("DELETE FROM event WHERE event_id = ANY(%s)", (a.generated_event_ids,))


def test_skew_window_aborts_batch(conn: Any) -> None:
    """An out-of-window event aborts the whole batch: the valid
    event is not written."""
    eid_valid = _uid()
    eid_invalid = _uid()
    old = NOW - timedelta(days=10)
    with pytest.raises(IngestValidationError):
        _ingest(
            conn,
            [_envelope(event_id=eid_valid), _envelope(event_id=eid_invalid, event_ts=old)],
        )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM event WHERE event_id = ANY(%s)",
            ([eid_valid, eid_invalid],),
        )
        assert int(cur.fetchone()[0]) == 0
