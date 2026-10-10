"""Unit tests for the readiness aggregate (ADR-0019)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from recsys.api.app import create_app
from recsys.api.readiness import aggregate
from recsys.config.enums import AppEnv
from recsys.config.settings import Settings


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"app_env": AppEnv.DEV}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- #
# aggregate — pure function
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("checks", "expected_status", "expected_code"),
    [
        (
            {
                "db": {"ok": True, "required": True},
                "index": {"ok": True, "required": True},
                "redis": {"ok": True, "required": False},
            },
            "ready",
            200,
        ),
        (
            {
                "db": {"ok": True, "required": True},
                "index": {"ok": True, "required": True},
                "redis": {"ok": False, "required": False, "error": "timeout"},
            },
            "degraded",
            200,
        ),
        (
            {
                "db": {"ok": False, "required": True, "error": "connection_refused"},
                "index": {"ok": True, "required": True},
                "redis": {"ok": True, "required": False},
            },
            "not_ready",
            503,
        ),
    ],
)
def test_aggregate(
    checks: dict[str, dict[str, Any]], expected_status: str, expected_code: int
) -> None:
    status, code = aggregate(checks)
    assert status == expected_status
    assert code == expected_code


# ---------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------- #
async def _async(value: Any) -> Any:
    return value


class _FakeOpenPool:
    """A pool that reports itself as open. The checks are
    monkeypatched, so it needs no connection()."""

    closed = False


def _patch_encoder(monkeypatch: pytest.MonkeyPatch, *, present: bool = True) -> None:
    """Patch the encoder builder.

    create_app stores the encoder in a closure variable and the
    lifespan assigns it to app.state. Setting app.state.encoder
    before TestClient enters the lifespan is overwritten by design
    (the same trap the pool fake fell into in test_recommend.py).
    """
    import recsys.api.app as app_module

    if present:
        monkeypatch.setattr(app_module, "_build_encoder", lambda _s, _l: object())
    else:
        monkeypatch.setattr(app_module, "_build_encoder", lambda _s, _l: None)


def _patch_health_checks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    db_ok: bool = True,
    index_ok: bool = True,
    redis_ok: bool = True,
) -> None:
    monkeypatch.setattr(
        "recsys.api.routers.health.check_db",
        lambda _pool, **_kw: {"ok": db_ok, "required": True, "latency_ms": 1},
    )
    monkeypatch.setattr(
        "recsys.api.routers.health.check_index",
        lambda _pool, **_kw: {
            "ok": index_ok,
            "required": True,
            "index_version": "idx-x",
            "row_count": 1,
            "latency_ms": 1,
        },
    )
    monkeypatch.setattr(
        "recsys.api.routers.health.check_redis",
        lambda _url, **_kw: _async({"ok": redis_ok, "required": False, "latency_ms": 1}),
    )


# ---------------------------------------------------------------- #
# /readyz
# ---------------------------------------------------------------- #
def test_readyz_reports_not_ready_without_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No database pool: db and index are not_ready, aggregate 503.

    The db pool builder is patched to a closed pool so the test is
    deterministic even when DATABASE_URL points at a live server
    (unit tests must not depend on external services; the check-
    markers gate enforces that).
    """
    import recsys.api.app as app_module

    class _ClosedPool:
        closed = True

    monkeypatch.setattr(app_module, "_build_db_pool", lambda _s: _ClosedPool())
    monkeypatch.setattr(
        "recsys.api.routers.health.check_redis",
        lambda _url, **_kw: _async({"ok": True, "required": False}),
    )
    app = create_app(_settings())
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["db"]["required"] is True
    assert body["checks"]["db"]["ok"] is False
    assert "encoder" in body["checks"]
    assert "redis" in body["checks"]


def test_readyz_degraded_when_redis_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Required checks pass; redis fails: degraded, 200."""
    _patch_health_checks(monkeypatch, redis_ok=False)
    _patch_encoder(monkeypatch, present=True)

    app = create_app(_settings())
    with TestClient(app) as c:
        app.state.db_pool = _FakeOpenPool()
        r = c.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["checks"]["db"]["ok"] is True
    assert body["checks"]["redis"]["ok"] is False


def test_readyz_ready_when_all_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_health_checks(monkeypatch)
    _patch_encoder(monkeypatch, present=True)

    app = create_app(_settings())
    with TestClient(app) as c:
        app.state.db_pool = _FakeOpenPool()
        r = c.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_readyz_encoder_not_required_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_health_checks(monkeypatch)
    # Encoder absent: its check reports ok=false but required=false
    # (dev); the aggregate is degraded, not not_ready.
    _patch_encoder(monkeypatch, present=False)

    app = create_app(_settings())
    with TestClient(app) as c:
        app.state.db_pool = _FakeOpenPool()
        r = c.get("/readyz")
    # Degraded (encoder optional and absent), not not_ready.
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["checks"]["encoder"]["required"] is False


# ---------------------------------------------------------------- #
# /livez and /healthz
# ---------------------------------------------------------------- #
def test_livez_returns_alive() -> None:
    app = create_app(_settings())
    with TestClient(app) as c:
        r = c.get("/livez")
    assert r.status_code == 200
    assert r.json() == {"status": "alive"}


def test_healthz_is_livez_alias() -> None:
    app = create_app(_settings())
    with TestClient(app) as c:
        a = c.get("/livez").json()
        b = c.get("/healthz").json()
    assert a == b == {"status": "alive"}
