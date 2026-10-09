"""Unit tests for deterministic experiment assignment (ADR-0017)."""

from __future__ import annotations

import datetime as dt

import pytest

from recsys.experiments.assignment import (
    BUCKET_SPACE,
    Assignment,
    assign,
    compute_bucket,
    effective_salt,
)
from recsys.experiments.loader import Experiment, ExperimentStatus


def _exp(
    *,
    name: str = "exp",
    salt: str = "s1",
    status: ExperimentStatus = ExperimentStatus.RUNNING,
    allocation: dict[str, int] | None = None,
) -> Experiment:
    return Experiment(
        name=name,
        salt=salt,
        status=status,
        allocation=allocation or {"control": 50, "treatment": 50},
        started_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        stopped_at=None
        if status is not ExperimentStatus.STOPPED
        else dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
        description="test",
    )


# ---------------------------------------------------------------- #
# effective_salt
# ---------------------------------------------------------------- #
def test_effective_salt_prefixes_env() -> None:
    assert effective_salt(experiment=_exp(salt="abc"), env="prod") == "prod:abc"


def test_effective_salt_override_replaces_env() -> None:
    s = effective_salt(experiment=_exp(salt="abc"), env="staging", env_override="prod")
    assert s == "prod:abc"


# ---------------------------------------------------------------- #
# compute_bucket
# ---------------------------------------------------------------- #
def test_bucket_in_range() -> None:
    b = compute_bucket(user_id="u_1", effective_salt_="s")
    assert 0 <= b < BUCKET_SPACE


def test_bucket_deterministic() -> None:
    a = compute_bucket(user_id="u_1", effective_salt_="s")
    b = compute_bucket(user_id="u_1", effective_salt_="s")
    assert a == b


def test_bucket_different_users_differ_in_general() -> None:
    # Not a distribution test: just that two distinct users do not
    # collide on the first try, which catches a hash that ignores
    # the user id.
    a = compute_bucket(user_id="u_1", effective_salt_="s")
    b = compute_bucket(user_id="u_2", effective_salt_="s")
    assert a != b


def test_bucket_different_salts_differ() -> None:
    a = compute_bucket(user_id="u_1", effective_salt_="s1")
    b = compute_bucket(user_id="u_1", effective_salt_="s2")
    assert a != b


def test_bucket_empty_user_id_raises() -> None:
    with pytest.raises(ValueError, match="user_id"):
        compute_bucket(user_id="", effective_salt_="s")


def test_bucket_empty_salt_raises() -> None:
    with pytest.raises(ValueError, match="effective_salt"):
        compute_bucket(user_id="u_1", effective_salt_="")


# ---------------------------------------------------------------- #
# assign — normal
# ---------------------------------------------------------------- #
def test_assign_running_returns_a_variant() -> None:
    a = assign(experiment=_exp(), user_id="u_1", env="prod")
    assert isinstance(a, Assignment)
    assert a.variant in {"control", "treatment"}
    assert 0 <= a.bucket < BUCKET_SPACE
    assert a.paused is False
    assert a.fallback_reason is None
    assert a.log_exposure is True


def test_assign_deterministic() -> None:
    a = assign(experiment=_exp(), user_id="u_1", env="prod")
    b = assign(experiment=_exp(), user_id="u_1", env="prod")
    assert a == b


def test_assign_allocation_maps_bucket() -> None:
    """A 30/70 split: the allocation is in percent, the bucket is in
    [0, BUCKET_SPACE). Buckets 0..2999 are control, 3000..9999 are
    treatment. The test asserts the mapping directly against the
    bucket value so it does not depend on a specific hash."""
    exp = _exp(allocation={"control": 30, "treatment": 70})
    boundary = BUCKET_SPACE * 30 // 100  # 3000
    seen = set()
    for i in range(500):
        a = assign(experiment=exp, user_id=f"u_{i}", env="prod")
        if a.bucket < boundary:
            assert a.variant == "control"
        else:
            assert a.variant == "treatment"
        seen.add(a.variant)
    # The 500-user sample must cover both arms on a 30/70 split;
    # otherwise the mapping is dropping one of them.
    assert seen == {"control", "treatment"}


def test_assign_env_prefix_changes_assignment() -> None:
    """A user is not guaranteed to land in the same variant across
    environments; the sample below must show at least one
    difference, which is what "independent assignments" means."""
    diffs = 0
    for i in range(200):
        a = assign(experiment=_exp(salt="abc"), user_id=f"u_{i}", env="dev")
        b = assign(experiment=_exp(salt="abc"), user_id=f"u_{i}", env="prod")
        if a.variant != b.variant:
            diffs += 1
    assert diffs > 0


def test_assign_override_reproduces_other_env() -> None:
    """Setting the override to the other environment produces the
    other environment's assignment for the same user."""
    for i in range(50):
        uid = f"u_{i}"
        prod = assign(experiment=_exp(), user_id=uid, env="prod")
        staging_with_override = assign(
            experiment=_exp(),
            user_id=uid,
            env="staging",
            env_override="prod",
        )
        assert prod == staging_with_override


# ---------------------------------------------------------------- #
# assign — paused
# ---------------------------------------------------------------- #
def test_paused_serves_control_and_logs() -> None:
    a = assign(
        experiment=_exp(status=ExperimentStatus.PAUSED),
        user_id="u_1",
        env="prod",
    )
    assert a.variant == "control"
    assert a.paused is True
    assert a.fallback_reason == "paused"
    assert a.log_exposure is True
    assert a.bucket >= 0


# ---------------------------------------------------------------- #
# assign — stopped
# ---------------------------------------------------------------- #
def test_stopped_serves_control_without_logging() -> None:
    a = assign(
        experiment=_exp(status=ExperimentStatus.STOPPED),
        user_id="u_1",
        env="prod",
    )
    assert a.variant == "control"
    assert a.paused is False
    assert a.fallback_reason == "stopped"
    assert a.log_exposure is False


# ---------------------------------------------------------------- #
# assign — disabled (kill switch)
# ---------------------------------------------------------------- #
def test_disabled_serves_control_without_logging() -> None:
    a = assign(experiment=_exp(), user_id="u_1", env="prod", disabled=True)
    assert a.variant == "control"
    assert a.paused is False
    assert a.fallback_reason == "disabled"
    assert a.log_exposure is False


def test_disabled_overrides_running() -> None:
    """The kill switch wins even when the experiment is running:
    that is the point of a kill switch."""
    a = assign(
        experiment=_exp(status=ExperimentStatus.RUNNING),
        user_id="u_1",
        env="prod",
        disabled=True,
    )
    assert a.fallback_reason == "disabled"


# ---------------------------------------------------------------- #
# control variant name
# ---------------------------------------------------------------- #
def test_control_is_first_allocation_key() -> None:
    """The fallback variant is the first key of the allocation,
    which by convention is ``control``. This lets an experiment
    name its arms without a special field, and the loader's
    schema is the same."""
    exp = _exp(allocation={"A": 50, "B": 50}, status=ExperimentStatus.PAUSED)
    a = assign(experiment=exp, user_id="u_1", env="prod")
    assert a.variant == "A"
