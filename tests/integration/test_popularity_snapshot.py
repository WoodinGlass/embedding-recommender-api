"""Integration test for the popularity snapshot refresh (ADR-0020).

Applies migrations (including 0003), seeds a small `item` table,
and runs the refresh against a real PostgreSQL. The refresh's
placeholder ranking (`ORDER BY md5(item_id)`) is deterministic, so
the tests can assert ordering.

Skipped unless ``RECSYS_TEST_DATABASE_URL`` is set.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator, Iterator
from typing import Any

import pytest

from recsys.popularity import refresh_popularity_snapshot

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
    with psycopg.connect(db_url) as c:
        yield c


@pytest.fixture
def seeded_items(conn: Any) -> Generator[list[str], None, None]:
    """Insert 10 items; return their ids.

    Uses a per-test prefix so two runs do not collide; the cleanup
    deletes them at the end. `popularity_snapshot` cascades on
    `item` delete, so the snapshot is cleaned too.
    """
    prefix = f"test-pop-{uuid.uuid4().hex[:8]}-"
    ids = [f"{prefix}{i:02d}" for i in range(10)]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO item (
                item_id, title, description, category, brand, language, content_hash
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [(iid, f"t-{iid}", f"d-{iid}", "cat", "brand", "en", f"h-{iid}") for iid in ids],
        )
    yield ids
    with conn.cursor() as cur:
        cur.execute("DELETE FROM item WHERE item_id = ANY(%s)", (ids,))


def _count(conn: Any) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM popularity_snapshot")
        return int(cur.fetchone()[0])


def _ids(conn: Any) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT item_id FROM popularity_snapshot ORDER BY rank ASC")
        return [str(r[0]) for r in cur.fetchall()]


def test_refresh_writes_size_rows(conn: Any, seeded_items: list[str]) -> None:
    written = refresh_popularity_snapshot(conn, size=5)
    assert written == 5
    assert _count(conn) == 5


def test_refresh_ranks_start_at_one(conn: Any, seeded_items: list[str]) -> None:
    refresh_popularity_snapshot(conn, size=5)
    with conn.cursor() as cur:
        cur.execute("SELECT rank FROM popularity_snapshot ORDER BY rank")
        ranks = [int(r[0]) for r in cur.fetchall()]
    assert ranks == [1, 2, 3, 4, 5]


def test_refresh_is_deterministic(conn: Any, seeded_items: list[str]) -> None:
    """Two refreshes with the same catalog and size produce the same
    ordered ids. The placeholder ranks by `md5(item_id)`; if it
    ranked by insertion order or by a random value, this would
    fail."""
    refresh_popularity_snapshot(conn, size=5)
    first = _ids(conn)
    refresh_popularity_snapshot(conn, size=5)
    second = _ids(conn)
    assert first == second


def test_refresh_replaces_snapshot(conn: Any, seeded_items: list[str]) -> None:
    """A smaller refresh removes the rows the larger one wrote; the
    table is replaced, not appended to."""
    refresh_popularity_snapshot(conn, size=5)
    assert _count(conn) == 5
    refresh_popularity_snapshot(conn, size=3)
    assert _count(conn) == 3


def test_refresh_with_size_larger_than_catalog(conn: Any, seeded_items: list[str]) -> None:
    """A `size` above the number of matching items writes every item
    and reports the smaller count; it does not fail."""
    written = refresh_popularity_snapshot(conn, size=100)
    assert written == 10
    assert _count(conn) == 10


def test_refresh_idempotent_on_same_size(conn: Any, seeded_items: list[str]) -> None:
    """Two refreshes in a row are fine; the second replaces the
    first without an error. This is the "run twice by accident"
    case."""
    a = refresh_popularity_snapshot(conn, size=5)
    b = refresh_popularity_snapshot(conn, size=5)
    assert a == b == 5
