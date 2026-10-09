"""Unit tests for configuration loading and validation.

The Settings model is the only place environment variables are read.
The tests below construct a ``Settings`` directly (bypassing ``.env``)
so the result does not depend on the developer's machine; the
``_env_file=None`` and the named overrides are the isolation mechanism
described in ADR-0022.

The prod guard is exercised for each of its rules: it is what makes a
misconfiguration a container-start failure instead of a first-request
failure.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import pytest
from pydantic import ValidationError

from recsys.config.enums import AppEnv, IndexBackend, LogFormat, LogLevel
from recsys.config.settings import Settings

pytestmark = pytest.mark.unit

#: A syntactically valid Argon2id hash used only in prod-guard tests.
#: The value does not need to verify; the guard checks the prefix.
_ARGON2_DUMMY = "$argon2id$v=19$m=65536,t=3,p=4$dummy$dummy"

#: A JWT secret long enough to pass the prod guard.
_LONG_SECRET = "x" * 40


def _settings(**overrides: Any) -> Settings:
    """Build Settings bypassing the .env file for test isolation.

    ``_env_file`` is a documented pydantic-settings runtime kwarg that
    the type stubs do not expose. ``Any`` for the overrides is the
    honest type at a **kwargs boundary that forwards to a validated
    constructor.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_defaults_are_dev_safe() -> None:
    s = _settings()
    assert s.app_env is AppEnv.DEV
    assert s.log_format is LogFormat.JSON
    assert s.log_level is LogLevel.INFO
    assert s.index_backend is IndexBackend.PGVECTOR
    assert s.metrics_enabled is True
    assert s.instance_count == 1
    assert s.jwt_max_age_seconds == 86_400
    assert s.user_id_hash_salt_version == 0
    assert s.latency_slo_ms == 200


def test_dev_defaults_do_not_require_secrets() -> None:
    s = _settings()
    assert s.api_keys == ""
    assert s.api_keys_admin == ""
    assert s.api_key_set == frozenset()
    assert s.api_key_admin_set == frozenset()


# --------------------------------------------------------------------------- #
# derived sets
# --------------------------------------------------------------------------- #
def test_api_key_set_ignores_whitespace_and_empties() -> None:
    # Semicolons separate entries; whitespace around an entry is trimmed.
    s = _settings(api_keys="  k1 ; k2 ;; k3  ")
    assert s.api_key_set == frozenset({"k1", "k2", "k3"})


def test_api_key_admin_set_parses() -> None:
    s = _settings(api_keys_admin="a1;a2")
    assert s.api_key_admin_set == frozenset({"a1", "a2"})


def test_empty_admin_keys_is_an_empty_set() -> None:
    s = _settings(api_keys_admin="")
    assert s.api_key_admin_set == frozenset()


# --------------------------------------------------------------------------- #
# field validators
# --------------------------------------------------------------------------- #
def test_ef_construction_must_be_at_least_m() -> None:
    with pytest.raises(ValidationError, match="HNSW_EF_CONSTRUCTION"):
        _settings(hnsw_m=32, hnsw_ef_construction=16)


def test_redis_breaker_open_max_must_be_at_least_open() -> None:
    with pytest.raises(ValidationError, match="REDIS_BREAKER_OPEN_MAX"):
        _settings(
            redis_breaker_open_seconds=10.0,
            redis_breaker_open_max_seconds=5.0,
        )


def test_db_breaker_open_max_must_be_at_least_open() -> None:
    with pytest.raises(ValidationError, match="DB_BREAKER_OPEN_MAX"):
        _settings(
            db_breaker_open_seconds=10.0,
            db_breaker_open_max_seconds=5.0,
        )


def test_jwt_algorithm_allowlist() -> None:
    # A disallowed algorithm is rejected.
    with pytest.raises(ValidationError, match="JWT_ALGORITHM"):
        _settings(jwt_algorithm="none")
    # HS256 and RS256 are accepted.
    assert _settings(jwt_algorithm="HS256").jwt_algorithm == "HS256"
    assert _settings(jwt_algorithm="RS256").jwt_algorithm == "RS256"


def test_cache_ttl_must_be_bounded() -> None:
    with pytest.raises(ValidationError):
        _settings(cache_ttl_seconds=-1)
    with pytest.raises(ValidationError):
        _settings(cache_ttl_seconds=86_401)


def test_readyz_cache_seconds_bounded() -> None:
    with pytest.raises(ValidationError):
        _settings(readyz_cache_seconds=61)


def test_otel_sampler_arg_bounded() -> None:
    with pytest.raises(ValidationError):
        _settings(otel_traces_sampler_arg=1.1)
    with pytest.raises(ValidationError):
        _settings(otel_traces_sampler_arg=-0.1)


def test_instance_count_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _settings(instance_count=0)


# --------------------------------------------------------------------------- #
# prod guard
# --------------------------------------------------------------------------- #
def _prod(**overrides: Any) -> Settings:
    """A minimal prod configuration; callers override what they test."""
    base: dict[str, Any] = {
        "app_env": AppEnv.PROD,
        "api_keys": _ARGON2_DUMMY,
        "jwt_secret": _LONG_SECRET,
        "user_id_hash_salt": "y" * 40,
        "user_id_hash_salt_version": 1,
    }
    base.update(overrides)
    return _settings(**base)


def test_prod_requires_api_keys() -> None:
    with pytest.raises(ValidationError, match="API_KEYS"):
        _prod(api_keys="")


def test_prod_rejects_plaintext_api_keys() -> None:
    with pytest.raises(ValidationError, match="Argon2id"):
        _prod(api_keys="plaintext-key")


def test_prod_rejects_default_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        _prod(jwt_secret="change-me-before-any-real-deployment")  # noqa: S106


def test_prod_rejects_short_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="32"):
        _prod(jwt_secret="short-secret")  # noqa: S106


def test_prod_rejects_default_user_id_hash_salt() -> None:
    with pytest.raises(ValidationError, match="USER_ID_HASH_SALT"):
        _prod(user_id_hash_salt="change-me-before-any-real-deployment")


def test_prod_rejects_zero_salt_version() -> None:
    with pytest.raises(ValidationError, match="SALT_VERSION"):
        _prod(user_id_hash_salt_version=0)


def test_prod_rejects_non_postgres_database_url() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        _prod(database_url="sqlite:///./data/app.sqlite")


def test_prod_rejects_non_redis_url() -> None:
    with pytest.raises(ValidationError, match="REDIS_URL"):
        _prod(redis_url="http://localhost:6379")


def test_prod_accepts_valid_config() -> None:
    s = _prod()
    assert s.app_env is AppEnv.PROD
    assert s.api_key_set == frozenset({_ARGON2_DUMMY})


def test_prod_allows_empty_admin_keys() -> None:
    s = _prod(api_keys_admin="")
    assert s.api_key_admin_set == frozenset()


# --------------------------------------------------------------------------- #
# import path
# --------------------------------------------------------------------------- #
def test_settings_does_not_import_the_api_layer() -> None:
    """The config layer must not import upward into the API.

    A Settings class that imported ``recsys.api.auth`` would create a
    cycle: the API imports Settings to know the secret; Settings would
    import the API to know the hash prefix. The prefix constant is
    duplicated in settings.py on purpose.

    The check is on import statements, not on the string ``recsys.api``
    anywhere in the file: a comment that mentions the API layer (which
    is where the reader needs to know why the constant is duplicated)
    is not an import.
    """
    import importlib
    import sys

    # Force a fresh import to inspect the module's dependencies.
    for name in list(sys.modules):
        if name == "recsys.config.settings" or name.startswith("recsys.config.settings."):
            del sys.modules[name]

    module = importlib.import_module("recsys.config.settings")
    module_file = module.__file__
    assert module_file is not None, "settings module has no __file__"
    source = pathlib.Path(module_file).read_text(encoding="utf-8")
    # Strip comments so the check reflects what the module does, not
    # what its docstrings and comments discuss.
    code_lines = []
    for line in source.splitlines():
        # Remove a trailing comment (anything after an unquoted #).
        # Adequate for this file: no # inside string literals.
        hash_index = line.find("#")
        if hash_index >= 0:
            line = line[:hash_index]
        code_lines.append(line)
    code = "\n".join(code_lines)
    forbidden = re.findall(
        r"^\s*(?:import|from)\s+recsys\.api\b",
        code,
        flags=re.MULTILINE,
    )
    assert not forbidden, (
        f"recsys.config.settings must not import from recsys.api; found: {forbidden}"
    )


# --------------------------------------------------------------------------- #
# Experiment Settings (ADR-0017)
# --------------------------------------------------------------------------- #
def test_experiment_disabled_default_false() -> None:
    # Memuat Settings dengan environment bersih; default harus False.
    import os
    from unittest.mock import patch

    env = {k: v for k, v in os.environ.items() if not k.startswith("EXPERIMENT_")}
    with patch.dict(os.environ, env, clear=True):
        s = Settings()
    assert s.experiment_disabled is False
    assert s.experiment_env_override is None


def test_experiment_env_override_accepts_known_envs() -> None:
    from unittest.mock import patch

    for value in ("dev", "staging", "prod"):
        with patch.dict("os.environ", {"EXPERIMENT_ENV_OVERRIDE": value}, clear=False):
            s = Settings()
            assert s.experiment_env_override == value


def test_experiment_env_override_rejects_unknown() -> None:
    from unittest.mock import patch

    import pytest

    with (
        patch.dict("os.environ", {"EXPERIMENT_ENV_OVERRIDE": "production"}, clear=False),
        pytest.raises(ValueError, match="EXPERIMENT_ENV_OVERRIDE"),
    ):
        Settings()
