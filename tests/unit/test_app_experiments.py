"""Unit tests for the experiments wiring in the app factory."""

from __future__ import annotations

from fastapi.testclient import TestClient

from recsys.config.enums import AppEnv
from recsys.config.settings import Settings
from recsys.experiments import ExperimentsFile


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"app_env": AppEnv.DEV}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_app_state_has_experiments() -> None:
    """create_app attaches the experiments declaration to app.state."""
    from recsys.api.app import create_app

    app = create_app(_settings())
    with TestClient(app):
        assert isinstance(app.state.experiments, ExperimentsFile)


def test_app_loads_real_experiments_yaml() -> None:
    """The committed experiments.yaml is loaded; its shape matches the
    loader's."""
    from recsys.api.app import create_app

    app = create_app(_settings())
    with TestClient(app):
        exp_file = app.state.experiments
        assert exp_file.schema_version == 1
        # The committed file ships one stopped example (ADR-0017).
        assert len(exp_file.experiments) >= 1
        names = {e.name for e in exp_file.experiments}
        assert "rerank_mmr" in names
