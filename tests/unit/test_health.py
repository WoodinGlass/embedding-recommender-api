"""Smoke tests for the app factory and health endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from recsys.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


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


def test_recommend_stub_returns_503(client: TestClient) -> None:
    r = client.post(
        "/v1/recommend",
        json={"user_id": "u_1", "seed_item_ids": ["i_1"], "k": 5},
    )
    assert r.status_code == 503
    assert "M3" in r.json()["detail"]


def test_recommend_rejects_unknown_filter(client: TestClient) -> None:
    r = client.post(
        "/v1/recommend",
        json={
            "user_id": "u_1",
            "seed_item_ids": ["i_1"],
            "k": 5,
            "filters": {"malicious_field": "x"},
        },
    )
    assert r.status_code == 422


def test_recommend_rejects_k_out_of_range(client: TestClient) -> None:
    r = client.post(
        "/v1/recommend",
        json={"user_id": "u_1", "seed_item_ids": [], "k": 999},
    )
    assert r.status_code == 422


def test_churn_stub_returns_503(client: TestClient) -> None:
    r = client.post("/v1/churn/score", json={"user_id": "u_1"})
    assert r.status_code == 503
    assert "M7" in r.json()["detail"]


def test_events_stub_returns_503(client: TestClient) -> None:
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
    assert r.status_code == 503
