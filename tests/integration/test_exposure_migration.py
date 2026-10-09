"""Integration test for the experiment_exposure table (ADR-0017).

Applies migration 0002 against a real PostgreSQL and exercises the
schema directly: the primary-key uniqueness (idempotency), the
bucket range check, the `user_id_hash` shape, and the
`fallback_reason` allowlist. The writer that fills the table lands
in M3.5.6; this file tests the table, not the writer.

Skipped unless ``RECSYS_TEST_DATABASE_URL`` is set. CI provides a
``pgvector/pgvector:pg16`` service container.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest

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
    with psycopg.connect(db_url, autocommit=True) as c:
        yield c


def _psycopg_errors() -> Any:
    """Return the `psycopg.errors` module, or skip."""
    psycopg = pytest.importorskip("psycopg")
    return psycopg.errors


def _row() -> tuple[str, str, str, str, str, int, int, str]:
    """Return a valid row's values.

    Tuple order: ``(event_id, experiment, variant, request_id,
    user_id_hash, user_id_hash_version, bucket, effective_salt)``.
    The hash is a fixed 64-hex string; the writer that produces real
    hashes is tested elsewhere.
    """
    return (
        uuid.uuid4().hex,
        "rerank_mmr",
        "control",
        uuid.uuid4().hex,
        "a" * 64,
        1,
        1234,
        "prod:2026-10-08-rerank-mmr-v1",
    )


def _insert(conn: Any, row: tuple[str, str, str, str, str, int, int, str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiment_exposure (
                event_id, experiment, variant, request_id,
                user_id_hash, user_id_hash_version, bucket, effective_salt
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (event_id) DO NOTHING
            """,
            row,
        )


def _insert_with_reason(
    conn: Any,
    row: tuple[str, str, str, str, str, int, int, str],
    reason: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO experiment_exposure (
                event_id, experiment, variant, request_id,
                user_id_hash, user_id_hash_version, bucket, effective_salt,
                fallback_reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (*row, reason),
        )


def _count(conn: Any, event_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM experiment_exposure WHERE event_id = %s",
            (event_id,),
        )
        return int(cur.fetchone()[0])


def test_insert_and_read_back(conn: Any) -> None:
    row = _row()
    _insert(conn, row)
    assert _count(conn, row[0]) == 1


def test_idempotent_on_event_id(conn: Any) -> None:
    """A second insert of the same `event_id` is a no-op. That is
    what makes a retried request safe (ADR-0017 § Exposure is
    idempotent by `request_id`)."""
    row = _row()
    _insert(conn, row)
    _insert(conn, row)
    assert _count(conn, row[0]) == 1


def test_bucket_out_of_range_rejected(conn: Any) -> None:
    errors = _psycopg_errors()
    row = list(_row())
    row[6] = 10_000  # >= BUCKET_SPACE
    with pytest.raises(errors.CheckViolation):
        _insert(conn, tuple(row))  # type: ignore[arg-type]


def test_bucket_negative_rejected(conn: Any) -> None:
    errors = _psycopg_errors()
    row = list(_row())
    row[6] = -1
    with pytest.raises(errors.CheckViolation):
        _insert(conn, tuple(row))  # type: ignore[arg-type]


def test_user_id_hash_must_be_64_lowercase_hex(conn: Any) -> None:
    errors = _psycopg_errors()
    row = list(_row())
    row[4] = "abc"  # too short
    with pytest.raises(errors.CheckViolation):
        _insert(conn, tuple(row))  # type: ignore[arg-type]


def test_user_id_hash_rejects_uppercase(conn: Any) -> None:
    errors = _psycopg_errors()
    row = list(_row())
    row[4] = "A" * 64  # uppercase is outside the regex
    with pytest.raises(errors.CheckViolation):
        _insert(conn, tuple(row))  # type: ignore[arg-type]


def test_fallback_reason_allowlist_rejects_unknown(conn: Any) -> None:
    errors = _psycopg_errors()
    row = _row()
    with pytest.raises(errors.CheckViolation):
        _insert_with_reason(conn, row, "not_a_real_reason")


def test_fallback_reason_accepts_known_values(conn: Any) -> None:
    for reason in ("paused", "stopped", "disabled", "allocation_gap"):
        row = _row()
        _insert_with_reason(conn, row, reason)
        assert _count(conn, row[0]) == 1


def test_paused_default_is_false(conn: Any) -> None:
    row = _row()
    _insert(conn, row)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT paused FROM experiment_exposure WHERE event_id = %s",
            (row[0],),
        )
        assert cur.fetchone()[0] is False
