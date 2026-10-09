"""Integration tests for auth wiring on every protected endpoint (M3.6.6b.3).

The unit tests in ``tests/unit/test_auth_dependency.py`` cover the
dependency against a minimal FastAPI app. This file covers the same
rule against the *real* app: the middleware stack, the routers, and
the app factory. A request without a credential is rejected before
the handler runs, so the 401 path does not need a live database; it
only needs the app to build.

The "credential accepted" case asserts *not 401*. The handler may
still return 400, 404, or 503 depending on state — those are the
handler's concerns, not auth's. Pinning the accepted case to 200
would couple this test to a working encoder, a live index, and a
seeded catalog, none of which are what this test is about.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from recsys.api.app import create_app
from recsys.config.enums import AppEnv
from recsys.config.settings import Settings

pytestmark = [pytest.mark.integration]

_TEST_API_KEY = "integration-test-key"


@pytest.fixture
def app_settings() -> Settings:
    """A dev Settings with one API key. Skips without a test database,
    matching the other integration tests' skip discipline."""
    if not os.environ.get("RECSYS_TEST_DATABASE_URL"):
        pytest.skip("set RECSYS_TEST_DATABASE_URL to run app-level integration tests")
    return Settings(app_env=AppEnv.DEV, api_keys=_TEST_API_KEY)


# ---------------------------------------------------------------- #
# request shapes — minimal valid bodies per endpoint
# ---------------------------------------------------------------- #
_RECOMMEND_BODY: dict[str, Any] = {"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 2}
_EVENTS_BODY: dict[str, Any] = {
    "event_id": "evt_0001",
    "event_ts": "2026-01-15T10:30:00Z",
    "event_type": "impression",
    "user_id": "u_1",
    "item_id": "i_1",
    "position": 1,
}
_CHURN_BODY: dict[str, Any] = {"user_id": "u_1"}

#: (method, path, body) for every endpoint that must require auth.
_PROTECTED: list[tuple[str, str, dict[str, Any] | None]] = [
    ("POST", "/v1/recommend", _RECOMMEND_BODY),
    ("GET", "/v1/items/i_1/similar", None),
    ("POST", "/v1/events", _EVENTS_BODY),
    ("POST", "/v1/churn/score", _CHURN_BODY),
]

#: (method, path) for every endpoint that must stay open.
_OPEN: list[tuple[str, str]] = [
    ("GET", "/healthz"),
    ("GET", "/readyz"),
    ("GET", "/metrics"),
]


def _send(client: TestClient, method: str, path: str, body: dict[str, Any] | None) -> Any:
    if method == "GET":
        return client.get(path)
    return client.post(path, json=body)


# ---------------------------------------------------------------- #
# 401 without a credential
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(("method", "path", "body"), _PROTECTED)
def test_no_credential_returns_401(
    app_settings: Settings, method: str, path: str, body: dict[str, Any] | None
) -> None:
    with TestClient(create_app(app_settings)) as c:
        r = _send(c, method, path, body)
    assert r.status_code == 401, f"{method} {path} did not require auth"
    detail = r.json()["detail"]
    assert detail["code"] == "unauthenticated"
    assert detail["request_id"]  # RequestContextMiddleware set it


@pytest.mark.parametrize(("method", "path", "body"), _PROTECTED)
def test_invalid_credential_returns_401(
    app_settings: Settings, method: str, path: str, body: dict[str, Any] | None
) -> None:
    with TestClient(create_app(app_settings), headers={"X-API-Key": "not-a-real-key"}) as c:
        r = _send(c, method, path, body)
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "unauthenticated"


# ---------------------------------------------------------------- #
# valid credential passes the auth layer
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(("method", "path", "body"), _PROTECTED)
def test_valid_credential_is_not_401(
    app_settings: Settings, method: str, path: str, body: dict[str, Any] | None
) -> None:
    with TestClient(create_app(app_settings), headers={"X-API-Key": _TEST_API_KEY}) as c:
        r = _send(c, method, path, body)
    assert r.status_code != 401, (
        f"{method} {path} rejected a valid credential; "
        "the auth layer must accept it and let the handler decide"
    )


# ---------------------------------------------------------------- #
# open endpoints stay open
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(("method", "path"), _OPEN)
def test_open_endpoints_do_not_require_auth(app_settings: Settings, method: str, path: str) -> None:
    with TestClient(create_app(app_settings)) as c:
        r = c.get(path)
    assert r.status_code != 401, f"{method} {path} must stay open"
