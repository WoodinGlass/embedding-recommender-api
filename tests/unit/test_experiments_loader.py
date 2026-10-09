"""Unit tests for the experiments.yaml loader (ADR-0017)."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import pytest
import yaml

from recsys.experiments.loader import (
    Experiment,
    ExperimentsError,
    ExperimentsFile,
    ExperimentStatus,
    load_experiments,
)

VALID: dict[str, Any] = {
    "schema_version": 1,
    "experiments": [
        {
            "name": "rerank_mmr",
            "salt": "2026-10-08-rerank-mmr-v1",
            "status": "running",
            "allocation": {"control": 50, "treatment": 50},
            "started_at": "2026-10-08T00:00:00Z",
            "stopped_at": None,
            "description": "MMR vs blended-only.",
        }
    ],
}


def _write(tmp_path: pathlib.Path, data: dict[str, Any]) -> pathlib.Path:
    p = tmp_path / "experiments.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _mutate(fn: Any) -> dict[str, Any]:
    d = copy.deepcopy(VALID)
    fn(d)
    return d


# ---------------------------------------------------------------- #
# happy path
# ---------------------------------------------------------------- #
def test_load_valid(tmp_path: pathlib.Path) -> None:
    f = load_experiments(_write(tmp_path, VALID))
    assert isinstance(f, ExperimentsFile)
    assert f.schema_version == 1
    assert len(f.experiments) == 1
    exp = f.experiments[0]
    assert isinstance(exp, Experiment)
    assert exp.name == "rerank_mmr"
    assert exp.status is ExperimentStatus.RUNNING
    assert exp.allocation == {"control": 50, "treatment": 50}
    assert exp.stopped_at is None
    assert exp.started_at.tzinfo is not None


def test_by_name(tmp_path: pathlib.Path) -> None:
    f = load_experiments(_write(tmp_path, VALID))
    assert f.by_name("rerank_mmr") is not None
    assert f.by_name("does_not_exist") is None


def test_stopped_requires_stopped_at(tmp_path: pathlib.Path) -> None:
    data = _mutate(
        lambda d: d["experiments"][0].update(status="stopped", stopped_at="2026-10-09T00:00:00Z")
    )
    f = load_experiments(_write(tmp_path, data))
    assert f.experiments[0].status is ExperimentStatus.STOPPED
    assert f.experiments[0].stopped_at is not None


# ---------------------------------------------------------------- #
# file-level failures
# ---------------------------------------------------------------- #
def test_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ExperimentsError, match="not found"):
        load_experiments(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "experiments.yaml"
    p.write_text("not: [valid: yaml", encoding="utf-8")
    with pytest.raises(ExperimentsError, match="not valid YAML"):
        load_experiments(p)


def test_root_not_mapping(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "experiments.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ExperimentsError, match="root must be a mapping"):
        load_experiments(p)


def test_wrong_schema_version(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d.update(schema_version=2))
    with pytest.raises(ExperimentsError, match="schema_version"):
        load_experiments(_write(tmp_path, data))


def test_unknown_top_level_field(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d.update(extra="x"))
    with pytest.raises(ExperimentsError, match="unknown top-level"):
        load_experiments(_write(tmp_path, data))


def test_empty_experiments_list(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d.update(experiments=[]))
    with pytest.raises(ExperimentsError, match="non-empty list"):
        load_experiments(_write(tmp_path, data))


# ---------------------------------------------------------------- #
# entry-level failures
# ---------------------------------------------------------------- #
def test_missing_field(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d["experiments"][0].pop("salt"))
    with pytest.raises(ExperimentsError, match="missing field"):
        load_experiments(_write(tmp_path, data))


def test_unknown_entry_field(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d["experiments"][0].update(extra="x"))
    with pytest.raises(ExperimentsError, match="unknown field"):
        load_experiments(_write(tmp_path, data))


def test_duplicate_name(tmp_path: pathlib.Path) -> None:
    data = _mutate(
        lambda d: d["experiments"].append(
            {**copy.deepcopy(d["experiments"][0]), "salt": "different-salt"}
        )
    )
    with pytest.raises(ExperimentsError, match="duplicate experiment name"):
        load_experiments(_write(tmp_path, data))


def test_duplicate_salt(tmp_path: pathlib.Path) -> None:
    data = _mutate(
        lambda d: d["experiments"].append({**copy.deepcopy(d["experiments"][0]), "name": "other"})
    )
    with pytest.raises(ExperimentsError, match="duplicate salt"):
        load_experiments(_write(tmp_path, data))


def test_bad_status(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d["experiments"][0].update(status="live"))
    with pytest.raises(ExperimentsError, match="status"):
        load_experiments(_write(tmp_path, data))


def test_allocation_not_summing_to_100(tmp_path: pathlib.Path) -> None:
    data = _mutate(
        lambda d: d["experiments"][0].update(allocation={"control": 40, "treatment": 40})
    )
    with pytest.raises(ExperimentsError, match="sum to exactly 100"):
        load_experiments(_write(tmp_path, data))


def test_allocation_negative(tmp_path: pathlib.Path) -> None:
    data = _mutate(
        lambda d: d["experiments"][0].update(allocation={"control": 150, "treatment": -50})
    )
    with pytest.raises(ExperimentsError, match=r"\[0, 100\]"):
        load_experiments(_write(tmp_path, data))


def test_allocation_bool_rejected(tmp_path: pathlib.Path) -> None:
    data = _mutate(
        lambda d: d["experiments"][0].update(allocation={"control": True, "treatment": 99})
    )
    with pytest.raises(ExperimentsError, match="must be an integer"):
        load_experiments(_write(tmp_path, data))


def test_started_at_requires_timezone(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d["experiments"][0].update(started_at="2026-10-08T00:00:00"))
    with pytest.raises(ExperimentsError, match="timezone"):
        load_experiments(_write(tmp_path, data))


def test_stopped_at_must_be_null_when_running(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d["experiments"][0].update(stopped_at="2026-10-09T00:00:00Z"))
    with pytest.raises(ExperimentsError, match="stopped_at must be null"):
        load_experiments(_write(tmp_path, data))


def test_stopped_at_required_when_stopped(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d["experiments"][0].update(status="stopped", stopped_at=None))
    with pytest.raises(ExperimentsError, match="stopped_at is required"):
        load_experiments(_write(tmp_path, data))


def test_empty_description_rejected(tmp_path: pathlib.Path) -> None:
    data = _mutate(lambda d: d["experiments"][0].update(description="   "))
    with pytest.raises(ExperimentsError, match="description"):
        load_experiments(_write(tmp_path, data))
