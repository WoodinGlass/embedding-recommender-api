"""Integration smoke tests — verify external services are reachable.

This module is the first real integration test. It runs only when the
``RECSYS_TEST_DATABASE_URL`` / ``RECSYS_TEST_REDIS_URL`` env vars are set
(the fixtures skip otherwise). In CI, GitHub Actions supplies them via
service containers.

When this file passes, the heavier M2/M3 integration tests can trust that the
infrastructure they depend on is present.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_postgres_reachable(integration_db_url: str) -> None:
    """Connect to PostgreSQL and run a trivial query."""
    import psycopg

    with (
        psycopg.connect(integration_db_url, connect_timeout=5) as conn,
        conn.cursor() as cur,
    ):
        cur.execute("SELECT 1")
        row = cur.fetchone()
    assert row == (1,)


def test_pgvector_extension_available(integration_db_url: str) -> None:
    """The ``vector`` extension must exist; retrieval depends on it in M2."""
    import psycopg

    with (
        psycopg.connect(integration_db_url, connect_timeout=5) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        )
        row = cur.fetchone()
    assert row == ("vector",), (
        "pgvector extension not found; use image pgvector/pgvector:pg16 or "
        "run: CREATE EXTENSION vector;"
    )


def test_redis_reachable(integration_redis_url: str) -> None:
    """Connect to Redis and ping."""
    import redis

    client = redis.Redis.from_url(integration_redis_url, socket_timeout=5)
    try:
        assert client.ping() is True
    finally:
        client.close()
