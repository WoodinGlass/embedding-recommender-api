"""Prometheus metrics.

Names and labels are frozen by ``docs/contracts.md`` § 4.1. Do not rename
without an ADR.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# A dedicated registry (not the default) keeps test isolation clean and lets
# us expose exactly what we intend at /metrics.
REGISTRY = CollectorRegistry(auto_describe=True)

# Buckets include 0.2 s so the p95 < 200 ms target is directly measurable.
_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    1.0,
    2.0,
    5.0,
)

REQUEST_DURATION = Histogram(
    "recsys_request_duration_seconds",
    "HTTP request latency by route, source, and status.",
    labelnames=("route", "source", "status"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

REQUESTS_TOTAL = Counter(
    "recsys_requests_total",
    "HTTP requests by route, method, status, and data source.",
    labelnames=("route", "method", "status", "source"),
    registry=REGISTRY,
)

CACHE_REQUESTS = Counter(
    "recsys_cache_requests_total",
    "Cache lookups by result.",
    labelnames=("result",),  # hit | miss | bypass | error
    registry=REGISTRY,
)

FALLBACK_TOTAL = Counter(
    "recsys_fallback_total",
    "Fallback responses by reason.",
    labelnames=("reason",),  # db_down | ann_timeout | cold_start | index_missing
    registry=REGISTRY,
)

ERRORS_TOTAL = Counter(
    "recsys_errors_total",
    "Errors by bounded category.",
    labelnames=("type",),  # auth | validation | retrieval | cache | upstream | internal
    registry=REGISTRY,
)

EMBEDDING_DRIFT = Gauge(
    "recsys_embedding_drift_score",
    "Centroid cosine shift over recent queries vs reference window.",
    labelnames=("window",),
    registry=REGISTRY,
)

ACTIVE_INDEX_INFO = Gauge(
    "recsys_active_index_info",
    "Always 1; label carries the active index and model version.",
    labelnames=("index_version", "model_version"),
    registry=REGISTRY,
)

EXPERIMENT_EXPOSURES = Counter(
    "recsys_experiment_exposures_total",
    "Variant exposures actually served.",
    labelnames=("experiment", "variant"),
    registry=REGISTRY,
)

EXPERIMENT_EXPOSURE_ERRORS = Counter(
    "recsys_experiment_exposure_errors_total",
    "Exposure writes that raised, by exception type.",
    labelnames=("type",),
    registry=REGISTRY,
)


RATE_LIMIT_DEGRADED = Gauge(
    "recsys_rate_limit_degraded",
    "1 when the shared Redis rate limiter is unavailable.",
    registry=REGISTRY,
)

RATE_LIMIT_HITS = Counter(
    "recsys_rate_limit_hits_total",
    "Rate-limit decisions by class and result.",
    labelnames=("class_name", "result"),  # allowed | limited
    registry=REGISTRY,
)

RATE_LIMIT_REMAINING = Histogram(
    "recsys_rate_limit_remaining",
    "Tokens remaining at decision time, by class.",
    labelnames=("class_name",),
    buckets=(0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000),
    registry=REGISTRY,
)

RATE_LIMIT_LUA_ERRORS_TOTAL = Counter(
    "recsys_rate_limit_lua_errors_total",
    "Rate-limit Lua or Redis failures, by exception type.",
    labelnames=("type",),
    registry=REGISTRY,
)

CACHE_NEGATIVE_HITS = Counter(
    "recsys_cache_negative_hits_total",
    'Hits on a negative ("no such item") cache entry.',
    registry=REGISTRY,
)

CACHE_WRITE_ERRORS_TOTAL = Counter(
    "recsys_cache_write_errors_total",
    "Cache writes that failed, by bounded exception type.",
    labelnames=("type",),
    registry=REGISTRY,
)

#: State values the gauge reports. Kept as constants so the callback that
#: sets the gauge and the test that reads it cannot drift.
BREAKER_CLOSED: int = 0
BREAKER_HALF_OPEN: int = 1
BREAKER_OPEN: int = 2

CIRCUIT_BREAKER_STATE = Gauge(
    "recsys_circuit_breaker_state",
    "Breaker state: 0=closed, 1=half_open, 2=open.",
    labelnames=("name",),
    registry=REGISTRY,
)

CIRCUIT_BREAKER_STATE_CHANGES_TOTAL = Counter(
    "recsys_circuit_breaker_state_changes_total",
    "Breaker state transitions, by name and from/to states.",
    labelnames=("name", "from", "to"),
    registry=REGISTRY,
)

CIRCUIT_BREAKER_TRIPS_TOTAL = Counter(
    "recsys_circuit_breaker_trips_total",
    "Times the breaker transitioned to OPEN.",
    labelnames=("name", "reason"),  # threshold | probe_failed
    registry=REGISTRY,
)

# ------------------------------------------------------------------ #
# re-ranker (ADR-0016)
# ------------------------------------------------------------------ #
RERANK_DURATION = Histogram(
    "recsys_rerank_duration_seconds",
    "Time spent in the re-ranker, by experiment arm.",
    labelnames=("arm",),  # retrieval | blend | blend_mmr
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

RERANK_FAILURES_TOTAL = Counter(
    "recsys_rerank_failures_total",
    "A re-ranker step that raised, by step and bounded exception type.",
    labelnames=("step", "type"),  # step in {normalize, blend, mmr}
    registry=REGISTRY,
)

RERANK_SKIPPED_TOTAL = Counter(
    "recsys_rerank_skipped_total",
    "A re-ranker step skipped, by reason.",
    labelnames=("reason",),  # no_vectors | k_below_min | no_candidates
    registry=REGISTRY,
)

RERANK_SIGNAL_MISSING_TOTAL = Counter(
    "recsys_rerank_signal_missing_total",
    "A signal provider that raised or timed out and was treated as neutral.",
    labelnames=("signal",),  # popularity | recency
    registry=REGISTRY,
)

RERANK_MMR_ACTIVE_TOTAL = Counter(
    "recsys_rerank_mmr_active_total",
    "Requests where MMR ran, by active flag.",
    labelnames=("active",),  # true | false
    registry=REGISTRY,
)

# ------------------------------------------------------------------ #
# request lifecycle (ADR-0012 amendment, M3.6.6)
# ------------------------------------------------------------------ #
ENCODER_MISSING_TOTAL = Counter(
    "recsys_encoder_missing_total",
    "Requests that ran without an encoder, by environment.",
    labelnames=("env",),  # dev | staging | prod
    registry=REGISTRY,
)

THREAD_POOL_WAITING = Gauge(
    "recsys_thread_pool_waiting",
    "Coroutines waiting for a synchronous worker thread.",
    registry=REGISTRY,
)

THREAD_POOL_ACTIVE = Gauge(
    "recsys_thread_pool_active",
    "Synchronous worker threads currently running a request.",
    registry=REGISTRY,
)


# --------------------------------------------------------------------- #
# Active index freshness (ADR-0015 amendment)
# --------------------------------------------------------------------- #
ACTIVE_INDEX_STALENESS_SECONDS = Gauge(
    "recsys_active_index_staleness_seconds",
    "Seconds since the last successful active-index refresh.",
    registry=REGISTRY,
)

ACTIVE_INDEX_REFRESH_FAILURES_TOTAL = Counter(
    "recsys_active_index_refresh_failures_total",
    "Active-index refresh attempts that raised.",
    registry=REGISTRY,
)

CACHE_SKIP_TOTAL = Counter(
    "recsys_cache_skip_total",
    "Requests that bypassed the cache for a reason other than the breaker.",
    labelnames=("reason",),
    registry=REGISTRY,
)
