"""Guard rail: metric names defined in code must be documented in contracts.md.

This test is what makes "contract drift is treated as a bug" enforceable.
"""

from __future__ import annotations

import pathlib

import pytest

from recsys.monitoring import metrics

CONTRACTS_PATH = pathlib.Path(__file__).resolve().parents[2] / "docs" / "contracts.md"


def test_contracts_file_exists() -> None:
    assert CONTRACTS_PATH.is_file(), f"Missing contract file: {CONTRACTS_PATH}"


@pytest.mark.parametrize(
    "metric_name",
    [
        "recsys_request_duration_seconds",
        "recsys_requests_total",
        "recsys_cache_requests_total",
        "recsys_cache_write_errors_total",
        "recsys_circuit_breaker_trips_total",
        "recsys_rerank_mmr_active_total",
        "recsys_thread_pool_active",
        "recsys_thread_pool_waiting",
        "recsys_encoder_missing_total",
        "recsys_rerank_signal_missing_total",
        "recsys_rerank_skipped_total",
        "recsys_rerank_failures_total",
        "recsys_rerank_duration_seconds",
        "recsys_circuit_breaker_state_changes_total",
        "recsys_circuit_breaker_state",
        "recsys_cache_negative_hits_total",
        "recsys_fallback_total",
        "recsys_errors_total",
        "recsys_embedding_drift_score",
        "recsys_active_index_info",
        "recsys_experiment_exposures_total",
        "recsys_rate_limit_degraded",
        "recsys_rate_limit_lua_errors_total",
        "recsys_rate_limit_remaining",
        "recsys_rate_limit_hits_total",
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
    assert metrics.RATE_LIMIT_HITS is not None
    assert metrics.CACHE_NEGATIVE_HITS is not None
    assert metrics.CACHE_WRITE_ERRORS_TOTAL is not None
    assert metrics.CIRCUIT_BREAKER_STATE is not None
    assert metrics.CIRCUIT_BREAKER_STATE_CHANGES_TOTAL is not None
    assert metrics.CIRCUIT_BREAKER_TRIPS_TOTAL is not None
    assert metrics.RERANK_DURATION is not None
    assert metrics.RERANK_FAILURES_TOTAL is not None
    assert metrics.RERANK_SKIPPED_TOTAL is not None
    assert metrics.RERANK_SIGNAL_MISSING_TOTAL is not None
    assert metrics.RERANK_MMR_ACTIVE_TOTAL is not None
    assert metrics.ENCODER_MISSING_TOTAL is not None
    assert metrics.THREAD_POOL_WAITING is not None
    assert metrics.THREAD_POOL_ACTIVE is not None
    assert metrics.RATE_LIMIT_REMAINING is not None
    assert metrics.RATE_LIMIT_LUA_ERRORS_TOTAL is not None
