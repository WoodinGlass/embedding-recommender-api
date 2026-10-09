"""Unit tests for the readyz encoder check (M3.6.6a.2)."""

from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from recsys.api.app import create_app
from recsys.config.enums import AppEnv
from recsys.config.settings import Settings


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"app_env": AppEnv.DEV}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_readyz_reports_encoder_in_dev_with_no_artifact(tmp_path: pathlib.Path) -> None:
    """Dev with no ONNX artifact: encoder is missing but not
    required, so the instance is ready and the check says why."""
    missing = tmp_path / "no_such_onnx_dir"
    app = create_app(_settings(embedding_onnx_path=str(missing)))
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"]["encoder"]["ok"] is False
    assert body["checks"]["encoder"]["required"] is False
    assert body["checks"]["encoder"]["error"] == "artifact_missing_or_unusable"
