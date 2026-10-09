"""Unit tests for the ``get_principal`` FastAPI dependency (ADR-0013).

A minimal FastAPI app is built here rather than the full
``create_app``: the point is the dependency, not the recommendation
pipeline. The app has one route that echoes the principal's subject
and scopes, so a 200 carries a proof of which credential validated.
"""

from __future__ import annotations

import time
from typing import Any

import jwt as _jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from recsys.api.auth.api_key import hash_api_key
from recsys.api.deps import PrincipalDep
from recsys.config.enums import AppEnv
from recsys.config.settings import Settings

pytestmark = pytest.mark.unit


_TEST_API_KEY = "unit-test-key"
_TEST_ADMIN_KEY = "unit-admin-key"
_TEST_JWT_SECRET = "x" * 40


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "app_env": AppEnv.DEV,
        "api_keys": _TEST_API_KEY,
        "api_keys_admin": _TEST_ADMIN_KEY,
        "jwt_secret": _TEST_JWT_SECRET,
        "jwt_algorithm": "HS256",
        "jwt_max_age_seconds": 3600,
    }
    base.update(overrides)
    return Settings(**base)


def _client(settings: Settings) -> TestClient:
    """Return a TestClient for a minimal app that echoes the principal."""
    app = FastAPI()
    app.state.settings = settings

    @app.get("/whoami")
    def whoami(principal: PrincipalDep) -> dict[str, Any]:
        return {
            "subject": principal.subject,
            "scopes": sorted(principal.scopes),
            "kind": principal.kind.value,
        }

    return TestClient(app)


# ---------------------------------------------------------------- #
# API key
# ---------------------------------------------------------------- #
def test_api_key_valid_returns_200() -> None:
    with _client(_settings()) as c:
        r = c.get("/whoami", headers={"X-API-Key": _TEST_API_KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "api_key"
    assert body["scopes"] == []


def test_admin_api_key_grants_admin_scope() -> None:
    with _client(_settings()) as c:
        r = c.get("/whoami", headers={"X-API-Key": _TEST_ADMIN_KEY})
    assert r.status_code == 200
    assert r.json()["scopes"] == ["admin"]


def test_invalid_api_key_returns_401() -> None:
    with _client(_settings()) as c:
        r = c.get("/whoami", headers={"X-API-Key": "not-a-key"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "unauthenticated"


def test_argon2_hashed_key_validates_in_dev() -> None:
    hashed = hash_api_key(_TEST_API_KEY)
    with _client(_settings(api_keys=hashed)) as c:
        r = c.get("/whoami", headers={"X-API-Key": _TEST_API_KEY})
    assert r.status_code == 200


# ---------------------------------------------------------------- #
# JWT
# ---------------------------------------------------------------- #
def _make_jwt(sub: str = "user-1", scope: str = "") -> str:
    now = int(time.time())
    payload: dict[str, Any] = {"sub": sub, "iat": now, "exp": now + 600}
    if scope:
        payload["scope"] = scope
    return _jwt.encode(payload, _TEST_JWT_SECRET, algorithm="HS256")


def test_jwt_valid_returns_200() -> None:
    with _client(_settings()) as c:
        r = c.get("/whoami", headers={"Authorization": f"Bearer {_make_jwt()}"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "jwt"
    assert body["subject"] == "user-1"


def test_jwt_with_admin_scope() -> None:
    with _client(_settings()) as c:
        r = c.get(
            "/whoami",
            headers={"Authorization": f"Bearer {_make_jwt(scope='admin')}"},
        )
    assert r.status_code == 200
    assert r.json()["scopes"] == ["admin"]


def test_jwt_wrong_secret_returns_401() -> None:
    token = _jwt.encode(
        {"sub": "u", "iat": 1, "exp": 9_999_999_999},
        "different-secret-" + "x" * 32,
        algorithm="HS256",
    )
    with _client(_settings()) as c:
        r = c.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_jwt_expired_returns_401() -> None:
    now = int(time.time())
    token = _jwt.encode(
        {"sub": "u", "iat": now - 7200, "exp": now - 3600},
        _TEST_JWT_SECRET,
        algorithm="HS256",
    )
    with _client(_settings()) as c:
        r = c.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


# ---------------------------------------------------------------- #
# missing credential
# ---------------------------------------------------------------- #
def test_no_credential_returns_401() -> None:
    with _client(_settings()) as c:
        r = c.get("/whoami")
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "unauthenticated"


def test_empty_api_key_header_returns_401() -> None:
    with _client(_settings()) as c:
        r = c.get("/whoami", headers={"X-API-Key": ""})
    assert r.status_code == 401


def test_bearer_prefix_without_token_returns_401() -> None:
    with _client(_settings()) as c:
        r = c.get("/whoami", headers={"Authorization": "Bearer "})
    assert r.status_code == 401


# ---------------------------------------------------------------- #
# precedence: API key wins when both headers are present
# ---------------------------------------------------------------- #
def test_api_key_takes_precedence_over_jwt() -> None:
    with _client(_settings()) as c:
        r = c.get(
            "/whoami",
            headers={
                "X-API-Key": _TEST_API_KEY,
                "Authorization": f"Bearer {_make_jwt(sub='jwt-user')}",
            },
        )
    assert r.status_code == 200
    assert r.json()["kind"] == "api_key"
