"""Loader and validator for ``experiments.yaml`` (ADR-0017).

The file is read once at API startup and validated. A malformed
file aborts startup: the failure mode is "the process does not
start", not "the process serves traffic with an unintended
assignment". The fallback for a *paused* or *stopped* experiment is
a runtime concern; it is not the answer to a typo.

The loader is pure: it opens one file, validates it, and returns a
value. Nothing global is mutated, no clock is read (timestamps are
parsed and stored as ``datetime``; the caller decides what "now"
means), and no log is written. The caller (``create_app``) decides
how to react to a failure.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import yaml

#: Only this schema version is accepted. A migration is a code
#: change, not a silent upgrade; the check is what makes it so.
SUPPORTED_SCHEMA_VERSION: int = 1


class ExperimentsError(ValueError):
    """The experiments file is missing, malformed, or unsafe."""


class ExperimentStatus(StrEnum):
    """The three lifecycle states ADR-0017 defines."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass(frozen=True)
class Experiment:
    """One declared experiment.

    ``allocation`` maps variant name to integer percent; the sum is
    exactly 100 (validated). ``started_at`` is timezone-aware UTC;
    ``stopped_at`` is ``None`` unless ``status`` is ``STOPPED``.
    """

    name: str
    salt: str
    status: ExperimentStatus
    allocation: dict[str, int]
    started_at: _dt.datetime
    stopped_at: _dt.datetime | None
    description: str


@dataclass(frozen=True)
class ExperimentsFile:
    """The parsed, validated content of ``experiments.yaml``."""

    schema_version: int
    experiments: tuple[Experiment, ...]
    path: pathlib.Path

    def by_name(self, name: str) -> Experiment | None:
        """Return the experiment called ``name``, or ``None``."""
        for exp in self.experiments:
            if exp.name == name:
                return exp
        return None


_ALLOWED_EXPERIMENT_FIELDS = frozenset(
    {
        "name",
        "salt",
        "status",
        "allocation",
        "started_at",
        "stopped_at",
        "description",
    }
)


def _parse_iso8601(value: Any, *, field: str) -> _dt.datetime:
    """Parse an ISO 8601 string; require a timezone.

    A naive timestamp is rejected: the file is committed and read
    on machines that may be in different timezones, and a value
    without an offset means something different on each. The check
    is what makes the file portable.
    """
    if not isinstance(value, str) or not value:
        raise ExperimentsError(f"{field} must be a non-empty ISO 8601 string")
    try:
        # ``fromisoformat`` accepts 'Z' since 3.11.
        dt = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ExperimentsError(f"{field} is not ISO 8601: {value!r}") from exc
    if dt.tzinfo is None:
        raise ExperimentsError(f"{field} must include a timezone offset: {value!r}")
    return dt


def _validate_allocation(raw: Any, *, where: str) -> dict[str, int]:
    """Return a validated allocation map.

    Keys are non-empty strings; values are integers in ``[0, 100]``;
    the sum is exactly 100. A sum other than 100 fails at startup
    rather than at the first request that hits an uncovered bucket.
    """
    if not isinstance(raw, dict) or not raw:
        raise ExperimentsError(f"{where}: allocation must be a non-empty mapping")
    out: dict[str, int] = {}
    for variant, pct in raw.items():
        if not isinstance(variant, str) or not variant:
            raise ExperimentsError(f"{where}: variant name must be a non-empty string")
        if isinstance(pct, bool) or not isinstance(pct, int):
            raise ExperimentsError(f"{where}: allocation for {variant!r} must be an integer")
        if not 0 <= pct <= 100:
            raise ExperimentsError(
                f"{where}: allocation for {variant!r} must be in [0, 100], got {pct}"
            )
        out[variant] = pct
    total = sum(out.values())
    if total != 100:
        raise ExperimentsError(f"{where}: allocation must sum to exactly 100, got {total}")
    return out


def _validate_experiment(raw: Any, *, where: str) -> Experiment:
    if not isinstance(raw, dict):
        raise ExperimentsError(f"{where}: experiment entry must be a mapping")
    unknown = set(raw.keys()) - _ALLOWED_EXPERIMENT_FIELDS
    if unknown:
        raise ExperimentsError(
            f"{where}: unknown field(s) {sorted(unknown)}; "
            f"allowed: {sorted(_ALLOWED_EXPERIMENT_FIELDS)}"
        )
    missing = _ALLOWED_EXPERIMENT_FIELDS - set(raw.keys())
    if missing:
        raise ExperimentsError(f"{where}: missing field(s) {sorted(missing)}")

    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise ExperimentsError(f"{where}: name must be a non-empty string")

    salt = raw["salt"]
    if not isinstance(salt, str) or not salt:
        raise ExperimentsError(f"{where}: salt must be a non-empty string")

    raw_status = raw["status"]
    try:
        status = ExperimentStatus(raw_status)
    except ValueError as exc:
        allowed = sorted(s.value for s in ExperimentStatus)
        raise ExperimentsError(
            f"{where}: status must be one of {allowed}, got {raw_status!r}"
        ) from exc

    allocation = _validate_allocation(raw["allocation"], where=where)

    started_at = _parse_iso8601(raw["started_at"], field=f"{where}.started_at")

    raw_stopped = raw["stopped_at"]
    if status is ExperimentStatus.STOPPED:
        if raw_stopped is None:
            raise ExperimentsError(f"{where}: stopped_at is required when status is 'stopped'")
        stopped_at = _parse_iso8601(raw_stopped, field=f"{where}.stopped_at")
    else:
        if raw_stopped is not None:
            raise ExperimentsError(
                f"{where}: stopped_at must be null when status is {status.value!r}"
            )
        stopped_at = None

    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ExperimentsError(f"{where}: description must be a non-empty string")

    return Experiment(
        name=name,
        salt=salt,
        status=status,
        allocation=allocation,
        started_at=started_at,
        stopped_at=stopped_at,
        description=description,
    )


def load_experiments(path: pathlib.Path) -> ExperimentsFile:
    """Read and validate ``experiments.yaml``. Raises ``ExperimentsError``.

    Checks, all from ADR-0017:

    - the root is a mapping with ``schema_version == 1``;
    - ``experiments`` is a non-empty list;
    - ``name`` is unique and non-empty, ``salt`` is unique and non-empty;
    - ``status`` is one of the three enum values;
    - ``allocation`` sums to exactly 100, keys non-empty, values int in [0, 100];
    - ``started_at`` is ISO 8601 with a timezone;
    - ``stopped_at`` is required iff ``status`` is ``stopped``;
    - unknown fields are rejected at both levels.
    """
    if not path.is_file():
        raise ExperimentsError(f"experiments file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExperimentsError(f"experiments file is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ExperimentsError("experiments file root must be a mapping")

    schema_version = raw.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ExperimentsError(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION}, got {schema_version!r}"
        )

    top_level_allowed = {"schema_version", "experiments"}
    unknown_top = set(raw.keys()) - top_level_allowed
    if unknown_top:
        raise ExperimentsError(
            f"unknown top-level field(s) {sorted(unknown_top)}; "
            f"allowed: {sorted(top_level_allowed)}"
        )

    raw_list = raw.get("experiments")
    if not isinstance(raw_list, list) or not raw_list:
        raise ExperimentsError("experiments must be a non-empty list")

    experiments: list[Experiment] = []
    seen_names: set[str] = set()
    seen_salts: set[str] = set()
    for i, entry in enumerate(raw_list):
        exp = _validate_experiment(entry, where=f"experiments[{i}]")
        if exp.name in seen_names:
            raise ExperimentsError(f"duplicate experiment name: {exp.name!r}")
        if exp.salt in seen_salts:
            raise ExperimentsError(
                f"duplicate salt {exp.salt!r} on {exp.name!r}; "
                "two experiments with the same salt are correlated"
            )
        seen_names.add(exp.name)
        seen_salts.add(exp.salt)
        experiments.append(exp)

    return ExperimentsFile(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        experiments=tuple(experiments),
        path=path,
    )


__all__ = [
    "SUPPORTED_SCHEMA_VERSION",
    "Experiment",
    "ExperimentStatus",
    "ExperimentsError",
    "ExperimentsFile",
    "load_experiments",
]
