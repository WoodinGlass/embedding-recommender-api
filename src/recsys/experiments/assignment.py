"""Deterministic experiment assignment (ADR-0017 § Decision).

The assignment is a pure function:

    bucket = sha256(f"{effective_salt}:{user_id}") % 10_000

where ``effective_salt = f"{env}:{experiment.salt}"`` and ``env`` is
``APP_ENV`` unless ``EXPERIMENT_ENV_OVERRIDE`` replaces it. The
bucket is then mapped to a variant by the experiment's allocation,
interpreted in declaration order (the order the YAML lists the
variants).

Why the environment prefix is prepended in code and not written in
the YAML: a promotion from staging to prod reuses the same ``salt``
string, and the two assignments are independent because the
environment differs. Writing the environment in the YAML would
require editing the file on every promotion and would make any
drift between the two files invisible until someone reads both.

Determinism is required for the M5 analysis and the M2 evaluation
gate to be meaningful (ADR-0017 § Consequences). The function reads
no clock, opens no connection, writes no log, and depends only on
its arguments.

**Ambiguity in ADR-0017 that this module resolves.** The status
table says a ``stopped`` experiment logs no exposure, while the
fallback section says a fallback assignment (including a stopped
one) still logs an exposure with a reason. This module follows the
status table: ``stopped`` (and ``EXPERIMENT_DISABLED=1``, which
ADR-0017 says is treated as stopped) does not log. The table is more
specific about per-request behavior; the fallback section is a
summary of the "control is served" decision, not of the logging
policy. If the analysis later needs the stopped-window traffic, an
ADR amendment changes this module and the call site together.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from recsys.experiments.loader import Experiment, ExperimentStatus

#: The bucket space. ADR-0017 fixes it at 10 000, which is fine-
#: grained enough that a 1% allocation is 100 buckets and coarse
#: enough that the modulus is a single integer operation.
BUCKET_SPACE: int = 10_000


@dataclass(frozen=True)
class Assignment:
    """The outcome of one assignment decision.

    ``variant`` is the variant the user is served. ``bucket`` is the
    raw bucket number (0..9999), kept for the exposure event's debug
    field so a reviewer can reproduce the assignment from the log.

    ``effective_salt`` is the salt the bucket was computed with,
    after the environment prefix and the override. It is carried
    through so a report can describe which salt produced a
    particular assignment without re-deriving it.

    ``paused`` is true only when the experiment's status is
    ``paused``. It is distinct from ``fallback_reason``: a paused
    experiment is a temporary hold and its exposures are counted;
    a fallback due to a kill switch or an allocation gap is not the
    same event.

    ``fallback_reason`` is one of ``None`` (a normal assignment),
    ``"paused"``, ``"stopped"``, ``"disabled"``, or
    ``"allocation_gap"``.

    ``log_exposure`` is the caller's instruction: the exposure
    writer runs only when this is true. It is false for a stopped
    or disabled experiment (no traffic is being experimented on)
    and true otherwise, including for a pause and an allocation
    gap: those are still trials of the control arm and the analysis
    needs to see them.
    """

    experiment_name: str
    variant: str
    bucket: int
    effective_salt: str
    paused: bool
    fallback_reason: str | None
    log_exposure: bool


def effective_salt(
    *,
    experiment: Experiment,
    env: str,
    env_override: str | None = None,
) -> str:
    """Return the salt the bucket is computed with.

    ``env`` is ``APP_ENV`` (``dev`` / ``staging`` / ``prod``).
    ``env_override``, when not ``None``, replaces it: the one use
    case ADR-0017 names is a staging replay that must reproduce a
    prod user's assignment. The caller (``Settings``) validates the
    override against the enum, so this function does not re-check.
    """
    prefix = env_override if env_override is not None else env
    return f"{prefix}:{experiment.salt}"


def compute_bucket(*, user_id: str, effective_salt_: str) -> int:
    """Return ``sha256(f"{salt}:{user_id}") % 10_000``.

    The parameter is named with a trailing underscore to avoid
    shadowing the module-level ``effective_salt`` function; the
    value is what that function returned.
    """
    if not user_id:
        raise ValueError("user_id must be a non-empty string")
    if not effective_salt_:
        raise ValueError("effective_salt must be a non-empty string")
    digest = hashlib.sha256(f"{effective_salt_}:{user_id}".encode()).hexdigest()
    return int(digest, 16) % BUCKET_SPACE


def _variant_for_bucket(
    allocation: dict[str, int],
    bucket: int,
) -> tuple[str, str | None]:
    """Map ``bucket`` to a variant by cumulative allocation.

    ``allocation`` values are **percent** (the loader validates that
    they sum to 100); ``bucket`` is in ``[0, BUCKET_SPACE)``. The
    comparison multiplies the cumulative percent by
    ``BUCKET_SPACE // 100`` so the two scales match: a 30% arm is
    buckets 0..2999 at the default space, not buckets 0..29.

    The dict's iteration order is the YAML's declaration order
    (Python 3.7+ preserves insertion order, and the loader builds the
    dict from ``yaml.safe_load`` which also preserves it). Returns
    ``(variant, fallback_reason)``; the reason is ``"allocation_gap"``
    when the bucket is not covered, which can only happen with a
    malformed file that the loader already rejects — the branch is a
    safety net.
    """
    # ``BUCKET_SPACE`` is 10 000 and percent is an integer 0..100, so
    # the division is exact. If a future change makes ``BUCKET_SPACE``
    # not divisible by 100, the mapping becomes a rounding question
    # worth its own ADR; the test would catch the change as a failure,
    # not a silent misroute.
    scale = BUCKET_SPACE // 100
    cumulative = 0
    for variant, percent in allocation.items():
        cumulative += percent
        if bucket < cumulative * scale:
            return variant, None
    # Unreachable for a file the loader accepted (sum == 100). Kept as
    # a defensive fallback: a future schema change that allowed
    # sum < 100 would land here.
    control = next(iter(allocation), "control")
    return control, "allocation_gap"


def assign(
    *,
    experiment: Experiment,
    user_id: str,
    env: str,
    env_override: str | None = None,
    disabled: bool = False,
) -> Assignment:
    """Return the variant ``user_id`` is served for ``experiment``.

    The order of the checks matters: the kill switch overrides
    status (ADR-0017 § Fallback variant, case 2), and a paused or
    stopped experiment still computes a bucket so the exposure
    event can carry it. The variant in every fallback case is the
    experiment's ``control`` — the first key of its allocation — or
    the literal ``"control"`` when the allocation does not name one.

    See the module docstring for the stopped-vs-fallback logging
    ambiguity this function resolves in favor of the status table.
    """
    salt = effective_salt(experiment=experiment, env=env, env_override=env_override)
    bucket = compute_bucket(user_id=user_id, effective_salt_=salt)
    control = next(iter(experiment.allocation), "control")

    if disabled:
        return Assignment(
            experiment_name=experiment.name,
            variant=control,
            bucket=bucket,
            effective_salt=salt,
            paused=False,
            fallback_reason="disabled",
            log_exposure=False,
        )

    if experiment.status is ExperimentStatus.STOPPED:
        return Assignment(
            experiment_name=experiment.name,
            variant=control,
            bucket=bucket,
            effective_salt=salt,
            paused=False,
            fallback_reason="stopped",
            log_exposure=False,
        )

    if experiment.status is ExperimentStatus.PAUSED:
        return Assignment(
            experiment_name=experiment.name,
            variant=control,
            bucket=bucket,
            effective_salt=salt,
            paused=True,
            fallback_reason="paused",
            log_exposure=True,
        )

    # status is RUNNING
    variant, reason = _variant_for_bucket(experiment.allocation, bucket)
    return Assignment(
        experiment_name=experiment.name,
        variant=variant,
        bucket=bucket,
        effective_salt=salt,
        paused=False,
        fallback_reason=reason,
        log_exposure=True,
    )


__all__ = [
    "BUCKET_SPACE",
    "Assignment",
    "assign",
    "compute_bucket",
    "effective_salt",
]
