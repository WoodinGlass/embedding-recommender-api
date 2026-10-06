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

RATE_LIMIT_DEGRADED = Gauge(
    "recsys_rate_limit_degraded",
    "1 when the shared Redis rate limiter is unavailable.",
    registry=REGISTRY,
)
