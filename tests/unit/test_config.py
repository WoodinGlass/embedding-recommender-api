"""Unit tests for configuration loading and validation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from recsys.config.enums import AppEnv, IndexBackend, LogFormat, LogLevel
from recsys.config.settings import Settings


def _settings(**overrides: Any) -> Settings:
    """Build Settings bypassing the .env file for test isolation.

    ``_env_file`` is a documented pydantic-settings runtime kwarg that the
    type stubs do not expose, hence the ignore. Process env is still consulted
    for anything not passed explicitly.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_defaults_are_dev_safe() -> None:
    s = _settings()
    assert s.app_env is AppEnv.DEV
    assert s.log_format is LogFormat.JSON
    assert s.log_level is LogLevel.INFO
    assert s.index_backend is IndexBackend.PGVECTOR
    assert s.metrics_enabled is True


def test_prod_requires_api_keys() -> None:
    with pytest.raises(ValidationError, match="API_KEYS"):
        _settings(app_env=AppEnv.PROD)


def test_prod_rejects_default_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        _settings(app_env=AppEnv.PROD, api_keys="k1")


def test_prod_accepts_explicit_secrets() -> None:
    s = _settings(
        app_env=AppEnv.PROD,
        api_keys="k1,k2",
        jwt_secret="s3cr3t",  # noqa: S106 — test value, never a real secret
    )
    assert s.api_key_set == frozenset({"k1", "k2"})


def test_ef_construction_must_be_at_least_m() -> None:
    with pytest.raises(ValidationError, match="HNSW_EF_CONSTRUCTION"):
        _settings(hnsw_m=32, hnsw_ef_construction=16)


def test_api_key_set_ignores_whitespace_and_empties() -> None:
    s = _settings(api_keys="  k1 , k2 ,, k3  ")
    assert s.api_key_set == frozenset({"k1", "k2", "k3"})


def test_cache_ttl_must_be_bounded() -> None:
    with pytest.raises(ValidationError):
        _settings(cache_ttl_seconds=-1)
    with pytest.raises(ValidationError):
        _settings(cache_ttl_seconds=86_401)
