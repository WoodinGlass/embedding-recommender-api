"""Unit tests for the evaluation threshold gate.

The gate is a pure function of a threshold file and a report. The tests
pin the four checks from ADR-0010 § Staleness and version checks plus
the two things the gate deliberately does NOT check (ADR-0010 § What the
gate does not check).
"""

from __future__ import annotations

import datetime as dt
import pathlib
from typing import Any

import pytest
import yaml

from recsys.evaluation.thresholds import (
    GateResult,
    MetricFailure,
    ThresholdError,
    evaluate_gate,
    load_thresholds,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_thresholds(
    path: pathlib.Path,
    *,
    golden_set_version: str = "v1",
    absolute_floor: dict[str, dict[str, float]] | None = None,
    per_system: dict[str, dict[str, float | None]] | None = None,
    extra_toplevel: str | None = None,
) -> None:
    """Write a thresholds YAML file for tests.

    Built from a Python dict and dumped via yaml.safe_dump, so an empty
    mapping is written as ``{}`` (not as a bare key with a null value) and
    a null threshold is written as ``null``. Hand-rolled f-strings get both
    of those wrong, and the loader (correctly) rejects a null
    absolute_floor.
    """
    doc: dict[str, object] = {
        "schema_version": 1,
        "golden_set_version": golden_set_version,
    }
    if absolute_floor is not None:
        doc["absolute_floor"] = absolute_floor
    if per_system is not None:
        doc.update(per_system)
    text = yaml.safe_dump(doc, sort_keys=False)
    if extra_toplevel:
        text = text + extra_toplevel
    path.write_text(text, encoding="utf-8")


def _report(
    *,
    golden_set_version: str = "v1",
    commit: str = "abc123",
    created_at: str = "2026-10-08T12:00:00Z",
    systems: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "golden_set_version": golden_set_version,
        "commit": commit,
        "created_at": created_at,
        "systems": systems
        if systems is not None
        else {
            "pgvector_hnsw": {
                "metrics": {
                    "recall_at_10": 0.90,
                    "ndcg_at_10": 0.85,
                    "mrr": 0.88,
                    "ann_recall_vs_exact": 0.95,
                }
            }
        },
    }


# --------------------------------------------------------------------------- #
# load_thresholds
# --------------------------------------------------------------------------- #
def test_load_happy_path(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={"pgvector_hnsw": {"recall_at_10": 0.70}},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    assert t.golden_set_version == "v1"
    assert t.absolute_floor["pgvector_hnsw"]["recall_at_10"] == 0.70
    assert t.per_system["pgvector_hnsw"]["recall_at_10"] == 0.85


def test_load_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ThresholdError, match="not found"):
        load_thresholds(tmp_path / "nope.yaml")


def test_load_invalid_yaml(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text(": not yaml\n", encoding="utf-8")
    with pytest.raises(ThresholdError, match="not valid YAML"):
        load_thresholds(p)


def test_load_not_a_mapping(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ThresholdError, match="must be a YAML mapping"):
        load_thresholds(p)


def test_load_wrong_schema_version(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(p, per_system={"pgvector_hnsw": {"recall_at_10": 0.85}})
    text = p.read_text(encoding="utf-8").replace("schema_version: 1", "schema_version: 2")
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ThresholdError, match="schema_version"):
        load_thresholds(p)


def test_load_missing_golden_set_version(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text("schema_version: 1\npgvector_hnsw:\n  recall_at_10: 0.85\n")
    with pytest.raises(ThresholdError, match="golden_set_version"):
        load_thresholds(p)


def test_load_missing_absolute_floor(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text(
        'schema_version: 1\ngolden_set_version: "v1"\npgvector_hnsw:\n  recall_at_10: 0.85\n'
    )
    with pytest.raises(ThresholdError, match="absolute_floor"):
        load_thresholds(p)


def test_load_no_per_system_thresholds(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text('schema_version: 1\ngolden_set_version: "v1"\nabsolute_floor: {}\n')
    with pytest.raises(ThresholdError, match="no per-system thresholds"):
        load_thresholds(p)


def test_load_null_threshold_skipped(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"random": {"recall_at_10": 0.0, "ann_recall_vs_exact": None}},
    )
    # YAML nulls must be represented in the file, so we build the file by
    # hand.
    p.write_text(
        'schema_version: 1\ngolden_set_version: "v1"\nabsolute_floor: {}\n'
        "random:\n  recall_at_10: 0.0\n  ann_recall_vs_exact: null\n",
        encoding="utf-8",
    )
    t = load_thresholds(p)
    assert "ann_recall_vs_exact" not in t.per_system["random"]
    assert t.per_system["random"]["recall_at_10"] == 0.0


def test_load_rejects_non_numeric_threshold(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text(
        'schema_version: 1\ngolden_set_version: "v1"\nabsolute_floor: {}\n'
        "pgvector_hnsw:\n  recall_at_10: high\n",
        encoding="utf-8",
    )
    with pytest.raises(ThresholdError, match="must be a number or null"):
        load_thresholds(p)


# --------------------------------------------------------------------------- #
# evaluate_gate — golden set version
# --------------------------------------------------------------------------- #
def test_gate_passes_when_all_above(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={"pgvector_hnsw": {"recall_at_10": 0.70}},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    result = evaluate_gate(t, _report(), current_commit="abc123", now=None)
    assert result.passed is True
    assert result.failures == []
    assert result.reason is None


def test_gate_fails_on_golden_set_version_mismatch(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    result = evaluate_gate(t, _report(golden_set_version="v2"), current_commit="abc123", now=None)
    assert result.passed is False
    assert "golden_set_version mismatch" in (result.reason or "")


def test_gate_fails_on_commit_mismatch(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    result = evaluate_gate(t, _report(commit="old"), current_commit="new", now=None)
    assert result.passed is False
    assert "commit mismatch" in (result.reason or "")


def test_gate_fails_on_stale_report(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    report = _report(created_at="2026-10-08T00:00:00Z")
    now = dt.datetime(2026, 10, 9, 12, 0, 0, tzinfo=dt.UTC)
    result = evaluate_gate(t, report, current_commit="abc123", now=now, max_report_age_hours=24.0)
    assert result.passed is False
    assert "old" in (result.reason or "")


def test_gate_skips_freshness_when_now_none(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    # Ancient report, but no freshness check requested.
    result = evaluate_gate(
        t, _report(created_at="2000-01-01T00:00:00Z"), current_commit="abc123", now=None
    )
    assert result.passed is True


def test_gate_missing_created_at_fails_freshness(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    report = _report()
    del report["created_at"]
    now = dt.datetime(2026, 10, 8, 13, tzinfo=dt.UTC)
    result = evaluate_gate(t, report, current_commit="abc123", now=now)
    assert result.passed is False
    assert "created_at" in (result.reason or "")


# --------------------------------------------------------------------------- #
# evaluate_gate — metric thresholds
# --------------------------------------------------------------------------- #
def test_gate_fails_on_per_system_threshold(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={"pgvector_hnsw": {"recall_at_10": 0.70}},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.95}},
    )
    t = load_thresholds(p)
    # Report has recall 0.90, per-system requires 0.95.
    result = evaluate_gate(t, _report(), current_commit="abc123", now=None)
    assert result.passed is False
    assert len(result.failures) == 1
    f = result.failures[0]
    assert f.system == "pgvector_hnsw"
    assert f.metric == "recall_at_10"
    assert f.kind == "per_system"
    assert f.value == pytest.approx(0.90)
    assert f.threshold == pytest.approx(0.95)


def test_gate_fails_on_absolute_floor(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={"pgvector_hnsw": {"recall_at_10": 0.80}},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    report = _report(systems={"pgvector_hnsw": {"metrics": {"recall_at_10": 0.75}}})
    result = evaluate_gate(t, report, current_commit="abc123", now=None)
    assert result.passed is False
    kinds = {f.kind for f in result.failures}
    assert "absolute_floor" in kinds
    assert "per_system" in kinds


def test_gate_reports_multiple_failures(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={
            "pgvector_hnsw": {
                "recall_at_10": 0.95,
                "ndcg_at_10": 0.95,
            }
        },
    )
    t = load_thresholds(p)
    result = evaluate_gate(t, _report(), current_commit="abc123", now=None)
    assert result.passed is False
    metrics = sorted(f.metric for f in result.failures)
    assert metrics == ["ndcg_at_10", "recall_at_10"]


def test_gate_ignores_metric_not_in_thresholds(tmp_path: pathlib.Path) -> None:
    # The report has a metric the thresholds file does not mention. The
    # gate ignores it: a new metric is not an automatic failure.
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    report = _report(
        systems={
            "pgvector_hnsw": {
                "metrics": {
                    "recall_at_10": 0.90,
                    "new_metric": 0.0,  # would fail if it were checked
                }
            }
        }
    )
    result = evaluate_gate(t, report, current_commit="abc123", now=None)
    assert result.passed is True


def test_gate_ignores_system_not_in_thresholds(tmp_path: pathlib.Path) -> None:
    # A system in the report with no thresholds is skipped.
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    report = _report(
        systems={
            "pgvector_hnsw": {"metrics": {"recall_at_10": 0.90}},
            "future_system": {"metrics": {"recall_at_10": 0.0}},
        }
    )
    result = evaluate_gate(t, report, current_commit="abc123", now=None)
    assert result.passed is True


def test_gate_missing_systems_fails(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "t.yaml"
    _write_thresholds(
        p,
        absolute_floor={},
        per_system={"pgvector_hnsw": {"recall_at_10": 0.85}},
    )
    t = load_thresholds(p)
    report = _report()
    del report["systems"]
    result = evaluate_gate(t, report, current_commit="abc123", now=None)
    assert result.passed is False
    assert "systems" in (result.reason or "")


# --------------------------------------------------------------------------- #
# frozen result
# --------------------------------------------------------------------------- #
def test_gate_result_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    r = GateResult(passed=True)
    with pytest.raises(FrozenInstanceError):
        r.passed = False  # type: ignore[misc]


def test_metric_failure_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    f = MetricFailure(system="s", metric="m", value=0.1, threshold=0.2, kind="per_system")
    with pytest.raises(FrozenInstanceError):
        f.value = 0.9  # type: ignore[misc]
