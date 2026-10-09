"""Threshold loading and gate logic.

The gate is defined by ADR-0010. It compares an evaluation report against
``evaluation/thresholds.yaml`` and returns a pass/fail decision with the
list of failing metrics. The loader and the comparison are pure: no I/O
beyond reading the YAML and the report, no side effects, no logging.

What the gate checks (ADR-0010 § Staleness and version checks):

1. The report's ``golden_set_version`` matches the threshold file's.
2. The report's ``commit`` matches the current commit under test, when
   the caller supplies one.
3. The report's ``created_at`` is no older than ``max_report_age_hours``,
   when the caller supplies a reference time.
4. Every metric in the report is at or above both its per-system
   threshold and the corresponding absolute floor (when defined).

The gate does not evaluate a metric that appears in the report but not in
the threshold file, and does not require a metric that appears in the
threshold file but not in the report. The former would reject every new
metric until someone adds it to the file; the latter would silently pass
when a system stops producing a metric it used to. Both are surprises.
See ADR-0010 § What the gate does not check.
"""

from __future__ import annotations

import datetime as dt
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import yaml

_SCHEMA_VERSION = 1


class ThresholdError(Exception):
    """Raised when the threshold file or the report is malformed."""


@dataclass(frozen=True)
class Thresholds:
    """A parsed, validated threshold file.

    ``per_system`` and ``absolute_floor`` are the two maps the gate
    compares against. ``informational_systems`` is a set of system
    names that appear in the report but are **not** gated: their
    metrics are logged and reviewed but do not fail a build. A new
    evaluation arm lands here until a first measurement exists to set
    a meaningful threshold; a threshold of ``0.00`` in ``per_system``
    would be a gate that cannot fail, which is worse than no gate at
    all because it hides the absence.
    """

    schema_version: int
    golden_set_version: str
    absolute_floor: dict[str, dict[str, float]]
    per_system: dict[str, dict[str, float]]
    informational_systems: frozenset[str] = frozenset()
    #: Operational floors for informational systems: metrics that must be
    #: checked even though the system is not gated on quality. Used for
    #: `provider_missing_rate_*` on the rerank arms — an arm whose signal
    #: provider was unavailable is not the arm it claims to be, and its
    #: numbers would be misread. See ADR-0017 and the M3.4 -> M3.5
    #: incident in CHANGELOG.md.
    informational_floors: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricFailure:
    """One metric below its threshold or floor."""

    system: str
    metric: str
    value: float
    threshold: float
    kind: Literal["per_system", "absolute_floor", "informational_floor"]


@dataclass(frozen=True)
class GateResult:
    """The outcome of a gate evaluation."""

    passed: bool
    failures: list[MetricFailure] = field(default_factory=list)
    reason: str | None = None  # populated only when passed is False and no
    # metric failure (version/staleness check)


def _parse_informational_floors(raw: Any) -> dict[str, dict[str, float]]:
    """Parse the optional ``informational_floors`` mapping.

    An absent section is the empty dict. A present section is a
    mapping from system name to a mapping of metric->number, the same
    shape as ``absolute_floor``. The check is deliberately the same as
    ``absolute_floor`` so a reader who knows one knows the other.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ThresholdError("informational_floors must be a mapping")
    out: dict[str, dict[str, float]] = {}
    for sys_name, metrics in raw.items():
        if not isinstance(sys_name, str):
            raise ThresholdError(
                f"informational_floors: system name must be a string, got {sys_name!r}"
            )
        if not isinstance(metrics, dict):
            raise ThresholdError(
                f"informational_floors.{sys_name}: must be a mapping of metric->number"
            )
        out[sys_name] = _validate_metric_map(metrics, f"informational_floors.{sys_name}")
    return out


def load_thresholds(path: pathlib.Path) -> Thresholds:
    """Read and validate ``thresholds.yaml``.

    The file must declare ``schema_version: 1`` and a
    ``golden_set_version``. ``absolute_floor`` and ``per_system`` are the
    two threshold maps; the file may use a different top-level key name
    for the per-system map (the ADR uses the system name itself, e.g.
    ``pgvector_hnsw:``). Both are recognized: any top-level mapping whose
    values are mappings of string→number is treated as a per-system map,
    except the reserved keys.
    """
    if not path.is_file():
        raise ThresholdError(f"thresholds file not found: {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ThresholdError(f"thresholds file is not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise ThresholdError("thresholds file must be a YAML mapping")

    schema_version = doc.get("schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise ThresholdError(
            f"unsupported schema_version {schema_version!r}; expected {_SCHEMA_VERSION}"
        )

    golden_set_version = doc.get("golden_set_version")
    if not isinstance(golden_set_version, str) or not golden_set_version:
        raise ThresholdError("golden_set_version must be a non-empty string")

    raw_floor = doc.get("absolute_floor")
    if not isinstance(raw_floor, dict):
        raise ThresholdError("absolute_floor must be a mapping")
    # absolute_floor is nested: {system_name: {metric_name: number}}.
    absolute_floor: dict[str, dict[str, float]] = {}
    for sys_name, metrics in raw_floor.items():
        if not isinstance(sys_name, str):
            raise ThresholdError(f"absolute_floor: system name must be a string, got {sys_name!r}")
        if not isinstance(metrics, dict):
            raise ThresholdError(f"absolute_floor.{sys_name}: must be a mapping of metric→number")
        absolute_floor[sys_name] = _validate_metric_map(metrics, f"absolute_floor.{sys_name}")

    informational_systems = _parse_informational(doc.get("informational"))
    informational_floors = _parse_informational_floors(doc.get("informational_floors"))

    per_system: dict[str, dict[str, float]] = {}
    reserved = {
        "schema_version",
        "golden_set_version",
        "absolute_floor",
        "informational",
        "informational_floors",
    }
    for key, value in doc.items():
        if key in reserved:
            continue
        if not isinstance(key, str):
            raise ThresholdError(f"top-level key must be a string, got {key!r}")
        if not isinstance(value, dict):
            raise ThresholdError(f"{key}: must be a mapping of metric→number")
        per_system[key] = _validate_metric_map(value, key)

    if not per_system:
        raise ThresholdError("no per-system thresholds found")

    # A system named in both places is a mistake: the informational list
    # is exactly the set of systems that are not gated, so having a
    # threshold for one of them is confusing. Reject rather than guess
    # which the author meant.
    overlap = informational_systems & set(per_system)
    if overlap:
        raise ThresholdError(f"system(s) in both gate and informational: {sorted(overlap)}")

    return Thresholds(
        schema_version=_SCHEMA_VERSION,
        golden_set_version=golden_set_version,
        absolute_floor=absolute_floor,
        per_system=per_system,
        informational_systems=informational_systems,
        informational_floors=informational_floors,
    )


def _parse_informational(raw: Any) -> frozenset[str]:
    """Parse the optional ``informational`` list of system names.

    An absent section is the empty set (no system is informational).
    A present section must be a list of non-empty strings. A string
    instead of a list is rejected: a single-name shortcut looks like a
    typo for a list and is not worth the flexibility.
    """
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        raise ThresholdError("informational must be a list of system names")
    out: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str) or not entry:
            raise ThresholdError(f"informational entries must be non-empty strings, got {entry!r}")
        out.add(entry)
    return frozenset(out)


def _validate_metric_map(value: dict[Any, Any], where: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise ThresholdError(f"{where}: metric name must be a string, got {k!r}")
        if v is None:
            # A null threshold means "no threshold for this metric". Used
            # for baselines whose ANN fidelity is not meaningful (e.g.
            # random has no fidelity, popularity has no fidelity).
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ThresholdError(f"{where}.{k}: threshold must be a number or null, got {v!r}")
        out[k] = float(v)
    return out


def evaluate_gate(
    thresholds: Thresholds,
    report: dict[str, Any],
    *,
    current_commit: str | None = None,
    now: dt.datetime | None = None,
    max_report_age_hours: float | None = 24.0,
    required_systems: Sequence[str] | None = None,
) -> GateResult:
    """Compare a report to the thresholds. Pure function.

    ``current_commit``, when given, is the commit under test. The report's
    ``commit`` field must match. ``now`` and ``max_report_age_hours``
    control the freshness check; passing ``now=None`` skips it.

    ``required_systems``, when given, is a list of system names that must
    appear in the report. A required system that is missing is a failure:
    a silently skipped system is the same as a disabled gate. When
    ``None``, the report is not required to contain any particular
    system.

    See module docstring for what else is and is not checked.
    """
    # 1. Golden set version
    report_gsv = report.get("golden_set_version")
    if report_gsv != thresholds.golden_set_version:
        return GateResult(
            passed=False,
            reason=(
                f"golden_set_version mismatch: report={report_gsv!r}, "
                f"thresholds={thresholds.golden_set_version!r}"
            ),
        )

    # 2. Commit
    if current_commit is not None:
        report_commit = report.get("commit")
        if report_commit != current_commit:
            return GateResult(
                passed=False,
                reason=(
                    f"report commit mismatch: report={report_commit!r}, current={current_commit!r}"
                ),
            )

    # 3. Freshness
    if now is not None and max_report_age_hours is not None:
        created_at_raw = report.get("created_at")
        if not isinstance(created_at_raw, str):
            return GateResult(
                passed=False,
                reason="report is missing 'created_at' (needed for freshness check)",
            )
        try:
            created_at = dt.datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        except ValueError:
            return GateResult(
                passed=False,
                reason=f"report 'created_at' is not ISO 8601: {created_at_raw!r}",
            )
        age_hours = (now - created_at).total_seconds() / 3600.0
        if age_hours > max_report_age_hours:
            return GateResult(
                passed=False,
                reason=(f"report is {age_hours:.1f}h old; max is {max_report_age_hours:.1f}h"),
            )

    # 3b. Required systems present
    if required_systems is not None:
        systems_obj = report.get("systems")
        if isinstance(systems_obj, dict):
            missing = [s for s in required_systems if s not in systems_obj]
            if missing:
                return GateResult(
                    passed=False,
                    reason=(f"required system(s) missing from report: {sorted(missing)}"),
                )

    # 4. Metrics
    failures: list[MetricFailure] = []
    systems = report.get("systems")
    if not isinstance(systems, dict):
        return GateResult(
            passed=False,
            reason="report is missing a 'systems' mapping",
        )
    for system_name, system_data in systems.items():
        if not isinstance(system_data, dict):
            continue
        # Informational systems are logged, not gated on quality. See
        # the Thresholds docstring for why: a system without a measured
        # baseline cannot have a meaningful threshold, and a threshold
        # of 0.00 is a gate that cannot fail.
        #
        # They may still be gated on an *operational* metric via
        # `informational_floors`: a `provider_missing_rate_*` above the
        # floor means the arm's numbers were produced with a signal
        # missing, and the numbers do not mean what the arm's name
        # says. This is the check that would have caught the M3.4
        # provider bug (see CHANGELOG.md).
        if system_name in thresholds.informational_systems:
            metrics = system_data.get("metrics")
            if isinstance(metrics, dict):
                info_floors = thresholds.informational_floors.get(system_name, {})
                for metric_name, value in metrics.items():
                    if metric_name not in info_floors:
                        continue
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    value_f = float(value)
                    t = info_floors[metric_name]
                    if value_f > t:
                        failures.append(
                            MetricFailure(
                                system=system_name,
                                metric=metric_name,
                                value=value_f,
                                threshold=t,
                                kind="informational_floor",
                            )
                        )
            continue
        metrics = system_data.get("metrics")
        if not isinstance(metrics, dict):
            continue

        per_system = thresholds.per_system.get(system_name, {})
        floor = thresholds.absolute_floor.get(system_name, {})

        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            value_f = float(value)

            if metric_name in per_system:
                t = per_system[metric_name]
                if value_f < t:
                    failures.append(
                        MetricFailure(
                            system=system_name,
                            metric=metric_name,
                            value=value_f,
                            threshold=t,
                            kind="per_system",
                        )
                    )

            if metric_name in floor:
                t = floor[metric_name]
                if value_f < t:
                    failures.append(
                        MetricFailure(
                            system=system_name,
                            metric=metric_name,
                            value=value_f,
                            threshold=t,
                            kind="absolute_floor",
                        )
                    )

    if failures:
        return GateResult(passed=False, failures=failures)
    return GateResult(passed=True)


__all__ = [
    "GateResult",
    "MetricFailure",
    "ThresholdError",
    "Thresholds",
    "evaluate_gate",
    "load_thresholds",
]
