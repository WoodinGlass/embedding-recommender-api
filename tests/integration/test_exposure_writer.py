"""Integration test for the exposure writer against real PostgreSQL.

Applies the schema (migration 0002) and exercises the writer's
behavior that the fake cannot: the `ON CONFLICT DO NOTHING`
returning `rowcount == 0` on a retry, and the transaction rollback
that leaves the shared connection usable after a failed insert.

Skipped unless ``RECSYS_TEST_DATABASE_URL`` is set. CI provides a
``pgvector/pgvector:pg16`` service container.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from recsys.events import UserIdHash
from recsys.experiments.assignment import Assignment
from recsys.experiments.exposure import (
    ExposureWriteError,
    write_exposure,
)

pytestmark = [pytest.mark.integration]


@pytest.fixture
def db_url() -> str:
    url = os.environ.get("RECSYS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set RECSYS_TEST_DATABASE_URL to run postgres integration tests")
    return url


@pytest.fixture
def conn(db_url: str) -> Iterator[Any]:
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(db_url, autocommit=False) as c:
        yield c
        c.rollback()


def _assignment(*, name: str = "rerank_mmr") -> Assignment:
    return Assignment(
        experiment_name=name,
        variant="control",
        bucket=1234,
        effective_salt="prod:2026-10-08-rerank-mmr-v1",
        paused=False,
        fallback_reason=None,
        log_exposure=True,
    )


def _hash() -> UserIdHash:
    return UserIdHash(hex="a" * 64, version=1)


def _count(conn: Any, event_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM experiment_exposure WHERE event_id = %s",
            (event_id,),
        )
        return int(cur.fetchone()[0])


def test_inserted_once(conn: Any) -> None:
    r = write_exposure(
        connection=conn,
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id=uuid.uuid4().hex,
    )
    assert r.inserted is True
    assert _count(conn, r.event_id) == 1


def test_retry_is_duplicate(conn: Any) -> None:
    """Same `request_id` twice: the second is a no-op, not an error.
    This is the property ADR-0017 § Exposure is idempotent by
    `request_id` requires."""
    rid = uuid.uuid4().hex
    a = write_exposure(
        connection=conn,
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id=rid,
    )
    assert a.inserted is True
    b = write_exposure(
        connection=conn,
        assignment=_assignment(),
        user_id_hash=_hash(),
        request_id=rid,
    )
    assert b.inserted is False
    assert b.skipped_reason == "duplicate"
    assert a.event_id == b.event_id
    assert _count(conn, a.event_id) == 1


def test_log_exposure_false_does_not_touch_db(conn: Any) -> None:
    a = _assignment()
    a = Assignment(
        experiment_name=a.experiment_name,
        variant=a.variant,
        bucket=a.bucket,
        effective_salt=a.effective_salt,
        paused=False,
        fallback_reason="stopped",
        log_exposure=False,
    )
    r = write_exposure(
        connection=conn,
        assignment=a,
        user_id_hash=_hash(),
        request_id=uuid.uuid4().hex,
    )
    assert r.inserted is False
    assert r.skipped_reason == "log_exposure_false"
    assert _count(conn, r.event_id) == 0


def test_failed_insert_leaves_connection_usable(conn: Any) -> None:
    """A failure inside the writer (here: a bucket the schema
    rejects) rolls back, and the next statement on the same
    connection succeeds. Without the explicit transaction in the
    writer, the connection would be in the aborted state (the M3.4
    PgRecencyProvider bug)."""
    bad = Assignment(
        experiment_name="rerank_mmr",
        variant="control",
        bucket=999_999,  # outside the CHECK
        effective_salt="prod:salt",
        paused=False,
        fallback_reason=None,
        log_exposure=True,
    )
    with pytest.raises(ExposureWriteError):
        write_exposure(
            connection=conn,
            assignment=bad,
            user_id_hash=_hash(),
            request_id=uuid.uuid4().hex,
        )
    # The connection must still work: run a trivial query.
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
