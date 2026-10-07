"""Integration fixtures for the M2 tests.

Adds two fixtures on top of the shared ``tests/conftest.py``:

- ``pg_connection`` — a psycopg connection to the test database, with
  migrations applied. The connection is closed at the end of the session.
- ``clean_db`` — truncates the M2 tables between tests so that each test
  starts from a known state.

Both skip unless ``RECSYS_TEST_DATABASE_URL`` is set, which CI provides
via the pgvector service container. Colab does not set it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def pg_connection() -> Iterator[Any]:
    """A live psycopg connection with migrations applied.

    Skips unless ``RECSYS_TEST_DATABASE_URL`` is set. Runs
    ``alembic upgrade head`` once per session, which is idempotent thanks
    to the ``alembic_version`` table.
    """
    url = os.environ.get("RECSYS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set RECSYS_TEST_DATABASE_URL to run database tests")

    # Ensure migrations are applied. Subprocess keeps the alembic
    # environment self-contained and mirrors how an operator would run it.
    env = dict(os.environ)
    env["DATABASE_URL"] = url
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")

    import psycopg

    conn = psycopg.connect(url, autocommit=False)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def clean_db(pg_connection: Any) -> Iterator[None]:
    """Truncate M2 tables before and after each test.

    Order matters only in principle: ``TRUNCATE ... CASCADE`` handles the
    FK chain. The truncate is what makes each test start from a known
    state without a schema reset.
    """
    import psycopg

    assert isinstance(pg_connection, psycopg.Connection)

    # If the previous test failed inside a transaction, the connection is
    # in INERROR and any statement — including TRUNCATE — is refused with
    # InFailedSqlTransaction. Rolling back first restores a usable
    # connection so one failing test does not cascade to the rest of the
    # session.
    pg_connection.rollback()

    with pg_connection.cursor() as cur:
        cur.execute("TRUNCATE TABLE embedding, item, index_registry RESTART IDENTITY CASCADE")
    pg_connection.commit()
    try:
        yield
    finally:
        pg_connection.rollback()
        with pg_connection.cursor() as cur:
            cur.execute("TRUNCATE TABLE embedding, item, index_registry RESTART IDENTITY CASCADE")
        pg_connection.commit()
