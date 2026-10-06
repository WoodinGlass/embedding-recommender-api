"""Guard rail: metric names defined in code must be documented in contracts.md.

This test is what makes "contract drift is treated as a bug" enforceable.
"""

from __future__ import annotations

import pathlib

import pytest

from recsys.monitoring import metrics

CONTRACTS_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "docs" / "contracts.md"
)


def test_contracts_file_exists() -> None:
    assert CONTRACTS_PATH.is_file(), f"Missing contract file: {CONTRACTS_PATH}"


@pytest.mark.parametrize(
    "metric_name",
    [
        "recsys_request_duration_seconds",
        "recsys_requests_total",
        "recsys_cache_requests_total",
        "recsys_fallback_total",
        "recsys_errors_total",
        "recsys_embedding_drift_score",
        "recsys_active_index_info",
        "recsys_experiment_exposures_total",
        "recsys_rate_limit_degraded",
    ],
)
def test_metric_name_documented(metric_name: str) -> None:
    contracts = CONTRACTS_PATH.read_text(encoding="utf-8")
    assert metric_name in contracts, (
        f"Metric {metric_name!r} is not documented in docs/contracts.md"
    )


def test_metric_objects_exposed() -> None:
    assert metrics.REQUEST_DURATION is not None
    assert metrics.REQUESTS_TOTAL is not None
    assert metrics.ACTIVE_INDEX_INFO is not None
    assert metrics.RATE_LIMIT_DEGRADED is not None
