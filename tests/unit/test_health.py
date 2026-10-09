"""Smoke tests for the app factory and health endpoints."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from recsys.api.app import create_app
from recsys.config.enums import AppEnv
from recsys.config.settings import Settings

#: The API key the authed_client fixture sends.
_TEST_API_KEY = "test-api-key"


def _settings() -> Settings:
    return Settings(app_env=AppEnv.DEV, api_keys=_TEST_API_KEY)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Unauthenticated client.

    The `with` block runs the lifespan; without it, the app
    state (hot_config, db_pool, encoder, ...) is never set and
    every dependency that reads it raises AttributeError.

    This client sends no credential; use it for /healthz,
    /readyz, /metrics, and the tests that expect a 401.
    """
    with TestClient(create_app(_settings())) as c:
        yield c


@pytest.fixture
def authed_client() -> Iterator[TestClient]:
    """Client with a valid X-API-Key. Use for tests that exercise
    the endpoint's own validation, not the auth dependency."""
    with TestClient(create_app(_settings()), headers={"X-API-Key": _TEST_API_KEY}) as c:
        yield c


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_metrics_exposes_active_index_info(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "recsys_active_index_info" in r.text


def test_request_id_header_round_trip(client: TestClient) -> None:
    r = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_request_id_generated_when_absent(client: TestClient) -> None:
    r = client.get("/healthz")
    assert len(r.headers["X-Request-ID"]) == 16


def test_request_id_truncated_when_too_long(client: TestClient) -> None:
    r = client.get("/healthz", headers={"X-Request-ID": "x" * 200})
    assert len(r.headers["X-Request-ID"]) == 64


def test_churn_stub_returns_503(authed_client: TestClient) -> None:
    r = authed_client.post("/v1/churn/score", json={"user_id": "u_1"})
    assert r.status_code == 503
    assert "M7" in r.json()["detail"]


def test_churn_requires_auth(client: TestClient) -> None:
    """No credential: the auth dependency rejects before the handler."""
    r = client.post("/v1/churn/score", json={"user_id": "u_1"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "unauthenticated"


def test_events_stub_returns_503(authed_client: TestClient) -> None:
    r = authed_client.post(
        "/v1/events",
        json={
            "event_id": "evt_0001",
            "event_ts": "2026-01-15T10:30:00Z",
            "event_type": "impression",
            "user_id": "u_1",
            "item_id": "i_1",
            "position": 1,
        },
    )
    assert r.status_code == 503


def test_events_requires_auth(client: TestClient) -> None:
    """No credential: the auth dependency rejects before the handler."""
    r = client.post(
        "/v1/events",
        json={
            "event_id": "evt_0001",
            "event_ts": "2026-01-15T10:30:00Z",
            "event_type": "impression",
            "user_id": "u_1",
            "item_id": "i_1",
            "position": 1,
        },
    )
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "unauthenticated"


def test_recommend_rejects_unknown_filter(authed_client: TestClient) -> None:
    r = authed_client.post(
        "/v1/recommend",
        json={
            "user_id": "u_1",
            "seed_item_ids": ["i_seed"],
            "k": 2,
            "filters": {"nonsense": "x"},
        },
    )
    # extra="forbid" on RecommendFilters: 422 regardless of auth.
    assert r.status_code == 422


def test_recommend_rejects_k_out_of_range(authed_client: TestClient) -> None:
    r = authed_client.post(
        "/v1/recommend",
        json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 9999},
    )
    assert r.status_code == 422


def test_recommend_requires_auth(client: TestClient) -> None:
    """No credential: the auth dependency rejects before the handler runs."""
    r = client.post(
        "/v1/recommend",
        json={"user_id": "u_1", "seed_item_ids": ["i_seed"], "k": 2},
    )
    assert r.status_code == 401
