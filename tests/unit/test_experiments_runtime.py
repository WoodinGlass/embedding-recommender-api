"""Unit tests for run_experiments_for_request (ADR-0017).

``write_exposure`` is monkeypatched on the runtime module so the
test does not need a database. What is under test is the loop:
which experiments get an assignment, which get an exposure write,
and how a write failure is handled.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from recsys.experiments import runtime as runtime_module
from recsys.experiments.exposure import ExposureResult, ExposureWriteError
from recsys.experiments.loader import (
    SUPPORTED_SCHEMA_VERSION,
    Experiment,
    ExperimentsFile,
    ExperimentStatus,
)
from recsys.experiments.runtime import run_experiments_for_request

pytestmark = pytest.mark.unit


_NOW = _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _experiment(
    name: str = "exp1",
    *,
    status: ExperimentStatus = ExperimentStatus.RUNNING,
) -> Experiment:
    return Experiment(
        name=name,
        salt=f"salt-{name}",
        status=status,
        allocation={"control": 50, "treatment": 50},
        started_at=_NOW,
        stopped_at=_NOW if status is ExperimentStatus.STOPPED else None,
        description="",
    )


def _file(*experiments: Experiment) -> ExperimentsFile:
    return ExperimentsFile(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        experiments=experiments,
        path=__import__("pathlib").Path("experiments.yaml"),
    )


def _call(
    *,
    experiments: ExperimentsFile,
    user_id: str = "u_1",
    request_id: str = "req_1",
    env: str = "dev",
    disabled: bool = False,
) -> tuple[Any, ...]:
    return run_experiments_for_request(
        connection=object(),  # opaque; write_exposure is patched
        experiments=experiments,
        user_id=user_id,
        request_id=request_id,
        env=env,
        disabled=disabled,
        env_override=None,
        user_id_salt="salt",
        user_id_salt_version=1,
    )


def _patch_write(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raises: bool = False,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _stub(*, connection: Any, assignment: Any, user_id_hash: Any, request_id: str) -> Any:
        calls.append(
            {
                "experiment": assignment.experiment_name,
                "variant": assignment.variant,
                "request_id": request_id,
            }
        )
        if raises:
            raise ExposureWriteError(cause=RuntimeError("boom"))
        return ExposureResult(event_id="evt", inserted=True)

    monkeypatch.setattr(runtime_module, "write_exposure", _stub)
    return calls


# ---------------------------------------------------------------- #
# empty file
# ---------------------------------------------------------------- #
def test_empty_file_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_write(monkeypatch)
    result = _call(experiments=_file())
    assert result == ()
    assert calls == []


# ---------------------------------------------------------------- #
# stopped experiment: assignment but no write
# ---------------------------------------------------------------- #
def test_stopped_experiment_no_write(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_write(monkeypatch)
    result = _call(experiments=_file(_experiment(status=ExperimentStatus.STOPPED)))
    assert len(result) == 1
    assert result[0].variant == "control"
    assert result[0].log_exposure is False
    assert calls == []


# ---------------------------------------------------------------- #
# running experiment: assignment + write
# ---------------------------------------------------------------- #
def test_running_experiment_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_write(monkeypatch)
    result = _call(experiments=_file(_experiment()))
    assert len(result) == 1
    assert result[0].log_exposure is True
    assert len(calls) == 1
    assert calls[0]["experiment"] == "exp1"
    assert calls[0]["request_id"] == "req_1"


# ---------------------------------------------------------------- #
# disabled: no writes
# ---------------------------------------------------------------- #
def test_disabled_kills_all_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_write(monkeypatch)
    result = _call(experiments=_file(_experiment()), disabled=True)
    assert len(result) == 1
    assert result[0].log_exposure is False
    assert calls == []


# ---------------------------------------------------------------- #
# write failure is caught and does not stop the next experiment
# ---------------------------------------------------------------- #
def test_write_failure_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_write(monkeypatch, raises=True)
    result = _call(experiments=_file(_experiment("a"), _experiment("b")))
    # Two assignments; both writes attempted; both failed; loop continued.
    assert len(result) == 2
    assert len(calls) == 2


# ---------------------------------------------------------------- #
# order is declaration order
# ---------------------------------------------------------------- #
def test_assignments_in_declaration_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_write(monkeypatch)
    result = _call(experiments=_file(_experiment("first"), _experiment("second")))
    assert [a.experiment_name for a in result] == ["first", "second"]
