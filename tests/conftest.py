"""Shared pytest configuration and fixtures.

Test tiers are separated by directory so that ``pytest -m unit`` is fast and
deterministic:

- ``tests/unit``         — no external services, milliseconds
- ``tests/integration``  — needs PostgreSQL (pgvector) and/or Redis
- ``tests/load``         — Locust/k6 scenarios, run manually

Tier markers are applied automatically by directory. Integration fixtures
skip cleanly when their env vars are absent, so ``pytest`` is always safe to
run locally. See ``tests/README.md`` for the full convention.
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# marker registration
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """Register markers for ``--strict-markers``.

    Duplicated from ``pyproject.toml`` on purpose: when pytest is invoked
    with an alternate rootdir (e.g. a single test file by path), the ini file
    may not be picked up and markers would be rejected as unknown.
    """
    config.addinivalue_line("markers", "unit: fast tests with no external services")
    config.addinivalue_line("markers", "integration: needs PostgreSQL (pgvector) and/or Redis")
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "load: Locust/k6 scenarios, run manually")


# ---------------------------------------------------------------------------
# automatic tier markers
# ---------------------------------------------------------------------------
# NOTE: pluggy matches hook arguments by name against the hookspec, so the
# parameter must be called `config` — an underscore prefix breaks registration.
# ARG001 (unused `config`) is already ignored for tests/** in pyproject.toml.
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply tier markers based on directory.

    Keeps every test file free of ``@pytest.mark.unit`` boilerplate while
    making ``pytest -m unit`` and ``pytest -m integration`` behave as
    expected.
    """
    for item in items:
        parts = item.path.parts
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        elif "load" in parts:
            item.add_marker(pytest.mark.load)
        elif "unit" in parts:
            item.add_marker(pytest.mark.unit)


# ---------------------------------------------------------------------------
# integration fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def integration_db_url() -> str:
    """PostgreSQL DSN for integration tests, or skip.

    Set ``RECSYS_TEST_DATABASE_URL`` to enable. In CI this is provided by the
    ``pgvector/pgvector:pg16`` service container.
    """
    url = os.environ.get("RECSYS_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set RECSYS_TEST_DATABASE_URL to run database integration tests")
    return url


@pytest.fixture(scope="session")
def integration_redis_url() -> str:
    """Redis URL for integration tests, or skip.

    Set ``RECSYS_TEST_REDIS_URL`` to enable. In CI this is provided by the
    ``redis:7`` service container.
    """
    url = os.environ.get("RECSYS_TEST_REDIS_URL")
    if not url:
        pytest.skip("set RECSYS_TEST_REDIS_URL to run redis integration tests")
    return url
