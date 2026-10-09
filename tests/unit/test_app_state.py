"""Unit tests for the app-state wiring (M3.6.5).

These tests assert that the app factory builds and attaches the
shared objects. They do not assert that the pool opens: opening is
a readiness concern (ADR-0019), and the test environment may or may
not have a database.
"""

from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from recsys.api.app import create_app
from recsys.config.enums import AppEnv
from recsys.config.settings import Settings
from recsys.popularity import PopularityCache


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"app_env": AppEnv.DEV}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_app_state_has_db_pool() -> None:
    app = create_app(_settings())
    with TestClient(app):
        assert app.state.db_pool is not None


def test_app_state_has_popularity_cache() -> None:
    app = create_app(_settings())
    with TestClient(app):
        assert isinstance(app.state.popularity_cache, PopularityCache)


def test_popularity_cache_missing_file_is_empty(tmp_path: pathlib.Path) -> None:
    """A path that does not exist produces an empty cache, not a
    crash. A fresh checkout has never run ``make popularity-refresh``
    and must still boot."""
    missing = tmp_path / "definitely_missing.json"
    app = create_app(_settings(popularity_cache_path=str(missing)))
    with TestClient(app):
        assert app.state.popularity_cache.size == 0
