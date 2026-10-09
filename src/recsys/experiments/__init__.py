"""Experiment assignment and exposure logging (ADR-0017)."""

from recsys.experiments.assignment import (
    BUCKET_SPACE,
    Assignment,
    assign,
    compute_bucket,
    effective_salt,
)
from recsys.experiments.exposure import (
    ExposureResult,
    ExposureWriteError,
    write_exposure,
)
from recsys.experiments.loader import (
    Experiment,
    ExperimentsError,
    ExperimentsFile,
    ExperimentStatus,
    load_experiments,
)

__all__ = [
    "BUCKET_SPACE",
    "Assignment",
    "Experiment",
    "ExperimentStatus",
    "ExperimentsError",
    "ExperimentsFile",
    "ExposureResult",
    "ExposureWriteError",
    "assign",
    "compute_bucket",
    "effective_salt",
    "load_experiments",
    "write_exposure",
]
