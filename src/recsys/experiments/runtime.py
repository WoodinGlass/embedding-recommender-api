"""Per-request experiment assignment and exposure logging (ADR-0017).

The handler calls :func:`run_experiments_for_request` once per
served response. It walks the declared experiments, assigns the
user to a variant, and writes an exposure row for each experiment
whose assignment says ``log_exposure``.

Why a function and not inline code in the handler: the loop has
three concerns (assignment, exposure, metrics) that all need to
stay in one place. A handler that iterated experiments itself
would grow every time ADR-0017 grows a field.

Why synchronous: ``write_exposure`` is a single indexed insert.
The caller (the handler) already runs on a worker thread for the
retrieval path; this function is invoked from that same thread, so
no extra hop is paid. The exposure count is small (one row per
declared experiment) and the experiments file is a handful of
entries.
"""

from __future__ import annotations

from typing import Any

from recsys.events import hash_user_id
from recsys.experiments.assignment import Assignment, assign
from recsys.experiments.exposure import ExposureWriteError, write_exposure
from recsys.experiments.loader import ExperimentsFile
from recsys.monitoring.logging import get_logger

log = get_logger(__name__)


def run_experiments_for_request(
    *,
    connection: Any,
    experiments: ExperimentsFile,
    user_id: str,
    request_id: str,
    env: str,
    disabled: bool,
    env_override: str | None,
    user_id_salt: str,
    user_id_salt_version: int,
) -> tuple[Assignment, ...]:
    """Assign + log exposure for every declared experiment.

    Returns the assignments in declaration order. The handler reads
    ``assignments[0]`` for ``meta.experiment``; a deployment with
    more than one running experiment will need a contract change
    (the response carries one experiment; ADR-0017 does not yet
    define multi-experiment responses).

    An empty file (dev/test with no ``experiments.yaml``) returns an
    empty tuple and logs nothing. A file with experiments whose
    status is ``stopped`` still produces an ``Assignment`` (with
    ``log_exposure=False``); the caller sees the assignment, the
    writer does not touch the database.

    Errors from a single exposure write are caught and counted; the
    request proceeds. An exposure is data quality, not correctness
    (ADR-0017 § Exposure is logged). A failure on experiment A does
    not skip experiment B: each write is independent.
    """
    if not experiments.experiments:
        return ()

    assignments: list[Assignment] = []
    user_id_hash = hash_user_id(user_id, salt=user_id_salt, salt_version=user_id_salt_version)

    for experiment in experiments.experiments:
        assignment = assign(
            experiment=experiment,
            user_id=user_id,
            env=env,
            env_override=env_override,
            disabled=disabled,
        )
        assignments.append(assignment)

        if not assignment.log_exposure:
            continue

        try:
            result = write_exposure(
                connection=connection,
                assignment=assignment,
                user_id_hash=user_id_hash,
                request_id=request_id,
            )
        except ExposureWriteError as exc:
            from recsys.monitoring.metrics import EXPERIMENT_EXPOSURE_ERRORS

            EXPERIMENT_EXPOSURE_ERRORS.labels(type=type(exc.cause).__name__).inc()
            log.warning(
                "experiment.exposure.failed",
                experiment=assignment.experiment_name,
                variant=assignment.variant,
                error_type=type(exc.cause).__name__,
            )
            continue

        if result.inserted:
            from recsys.monitoring.metrics import EXPERIMENT_EXPOSURES

            EXPERIMENT_EXPOSURES.labels(
                experiment=assignment.experiment_name,
                variant=assignment.variant,
            ).inc()
            log.debug(
                "experiment.exposure.written",
                experiment=assignment.experiment_name,
                variant=assignment.variant,
            )

    return tuple(assignments)


__all__ = ["run_experiments_for_request"]
