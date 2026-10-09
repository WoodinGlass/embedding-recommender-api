"""Integration test for the tier-3 snapshot reader (ADR-0020)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from typing import Any

import pytest

from recsys.popularity import read_snapshot

pytestmark = [pytest.mark.integration]


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


@pytest.fixture
def seeded(conn: Any) -> Generator[list[str], None, None]:
    prefix = f"test-rd-{uuid.uuid4().hex[:8]}-"
    ids = [f"{prefix}{i:02d}" for i in range(5)]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO item (item_id, title, description, category, brand, language, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (iid, iid, iid, "cat_a" if i % 2 == 0 else "cat_b", "brand", "en", iid)
                for i, iid in enumerate(ids)
            ],
        )
        cur.executemany(
            """
            INSERT INTO popularity_snapshot (item_id, rank, category, brand, language)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (iid, i + 1, "cat_a" if i % 2 == 0 else "cat_b", "brand", "en")
                for i, iid in enumerate(ids)
            ],
        )
    yield ids
    with conn.cursor() as cur:
        cur.execute("DELETE FROM item WHERE item_id = ANY(%s)", (ids,))


def test_reads_rows_in_rank_order(conn: Any, seeded: list[str]) -> None:
    out = read_snapshot(conn, k=1000)
    ours = [iid for iid, _ in out if iid in seeded]
    assert ours == seeded


def test_limits_to_k(conn: Any, seeded: list[str]) -> None:
    out = read_snapshot(conn, k=2)
    assert len(out) == 2


def test_filter_by_category(conn: Any, seeded: list[str]) -> None:
    cat_a = [iid for i, iid in enumerate(seeded) if i % 2 == 0]
    out = read_snapshot(conn, k=1000, filters={"category": "cat_a"})
    returned = [iid for iid, _ in out if iid in seeded]
    assert set(returned) == set(cat_a)


def test_no_match_returns_empty(conn: Any, seeded: list[str]) -> None:
    out = read_snapshot(conn, k=10, filters={"category": "nonexistent_xyz"})
    assert out == []


def test_scores_in_open_closed_unit_interval(conn: Any, seeded: list[str]) -> None:
    out = read_snapshot(conn, k=1000)
    for _iid, score in out:
        assert 0.0 < score <= 1.0


def test_k_must_be_positive(conn: Any) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        read_snapshot(conn, k=0)
