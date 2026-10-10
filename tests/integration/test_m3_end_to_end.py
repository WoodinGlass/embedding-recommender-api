"""M3 end-to-end tests against a real app, real DB, real Redis.

The exit criterion for M3 is "integration tests are green". This
file is the assertion that the full serving stack — the app
factory, the middleware chain, auth, the connection pool, the
readiness checks, and the recommend handler — runs against the
real dependencies and produces the shapes the contract fixes.

It does not seed a catalog or build an index. That is the
evaluation gate's job. What it proves here is that the app boots
with a reachable database and Redis and answers with the right
envelope for the empty-catalog state.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from recsys.api.app import create_app
from recsys.config.enums import AppEnv
from recsys.config.settings import Settings

pytestmark = [pytest.mark.integration]

_TEST_API_KEY = "m3-e2e-key"


@pytest.fixture
def real_settings() -> Settings:
    if not os.environ.get("RECSYS_TEST_DATABASE_URL"):
        pytest.skip("set RECSYS_TEST_DATABASE_URL to run app-level integration tests")
    if not os.environ.get("RECSYS_TEST_REDIS_URL"):
        pytest.skip("set RECSYS_TEST_REDIS_URL to run app-level integration tests")
    return Settings(
        app_env=AppEnv.DEV,
        api_keys=_TEST_API_KEY,
        database_url=os.environ["RECSYS_TEST_DATABASE_URL"],
        redis_url=os.environ["RECSYS_TEST_REDIS_URL"],
    )


@pytest.fixture
def real_client(real_settings: Settings) -> Iterator[TestClient]:
    """A TestClient whose lifespan reaches the real DB and Redis."""
    app = create_app(real_settings)
    with TestClient(app, headers={"X-API-Key": _TEST_API_KEY}) as c:
        yield c


# ---------------------------------------------------------------- #
# readiness against real services
# ---------------------------------------------------------------- #
def test_readyz_reports_real_db_and_redis_ok(real_client: TestClient) -> None:
    """The readiness body's shape and the required/optional split.

    The database is a required check: if it is unreachable the test
    fails, because everything M3 serves depends on it. Redis is
    optional (ADR-0019): it may be up or down, and the body says
    which; a Redis hiccup degrades the instance, it does not fail
    the test. The encoder and index checks depend on what the
    environment has provisioned; the contract fixes their shape,
    not their value.
    """
    r = real_client.get("/readyz")
    body = r.json()
    assert body["checks"]["db"]["ok"] is True, "real database is unreachable"
    assert body["checks"]["db"]["required"] is True
    assert body["checks"]["redis"]["required"] is False
    for name in ("encoder", "index", "redis"):
        assert name in body["checks"], f"missing check: {name}"
        assert "ok" in body["checks"][name]


def test_readyz_returns_503_or_200_but_never_500(real_client: TestClient) -> None:
    """The aggregate is one of ready / degraded / not_ready. In a
    bare CI database (migrations applied, no active index, no
    catalog) it is not_ready because the index check fails; the
    handler must return 503 with the contract body, never 500."""
    r = real_client.get("/readyz")
    assert r.status_code in (200, 503)
    body = r.json()
    assert body["status"] in ("ready", "degraded", "not_ready")


# ---------------------------------------------------------------- #
# recommend: full stack, empty catalog
# ---------------------------------------------------------------- #
def test_recommend_with_no_active_index_returns_503(real_client: TestClient, clean_db: Any) -> None:
    """The pipeline returns ``no_active_index`` (no row is active),
    the fallback chain produces nothing, and the handler returns
    the 503 ``unavailable`` envelope. This exercises the whole
    stack: middleware, auth, pool, pipeline, fallback chain."""
    r = real_client.post(
        "/v1/recommend",
        json={"user_id": "u_1", "seed_item_ids": ["i_1"], "k": 2},
    )
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] == "unavailable"
    assert "no_active_index" in detail["message"]
    assert detail["request_id"]


def test_recommend_without_credential_returns_401(real_settings: Settings) -> None:
    """A fresh client with no header: 401, before any DB access."""
    app = create_app(real_settings)
    with TestClient(app) as c:
        r = c.post(
            "/v1/recommend",
            json={"user_id": "u_1", "seed_item_ids": ["i_1"], "k": 2},
        )
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "unauthenticated"


def test_events_accepts_batch_and_returns_202(real_client: TestClient) -> None:
    """The events handler runs against real DB and returns the ack
    shape the contract fixes (accepted / duplicates / rejected)."""
    r = real_client.post(
        "/v1/events",
        json={
            "events": [
                {
                    "event_id": "m3-e2e-evt-0001",
                    "event_ts": "2026-10-10T00:00:00Z",
                    "event_type": "impression",
                    "user_id": "u_1",
                    "item_id": "i_1",
                    "position": 1,
                }
            ]
        },
    )
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] >= 0
    assert body["duplicates"] >= 0
    assert body["rejected"] >= 0


def test_similar_with_no_active_index_returns_503(real_client: TestClient, clean_db: Any) -> None:
    """Same as the recommend case, for the item-scoped endpoint."""
    r = real_client.get("/v1/items/i_1/similar")
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "unavailable"
