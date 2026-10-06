# Contracts

> **Status:** locked. Producer and consumer code must conform to this document.
> A change to any contract requires a new ADR in `docs/adr/`.
> High-level decisions live in [`docs/decisions.md`](decisions.md).

This file is the single source of truth for **schemas** (data, API, config,
telemetry) and **idempotency keys** per endpoint. Pydantic models in
`src/recsys/` mirror these definitions; when they drift, this document wins
and the code is fixed.

---

## 0. Cross-cutting conventions

| Concern | Rule |
|---|---|
| JSON field names | `snake_case` everywhere (no `camelCase`, no exceptions) |
| Timestamps | ISO 8601 with `Z` suffix, UTC (e.g. `2026-01-15T10:30:00Z`) |
| Durations | Integer milliseconds in payloads, `float` seconds in metrics |
| IDs | Opaque strings. Consumers must not parse or assume structure. |
| Scores in retrieval | Float in `[0.0, 1.0]` (cosine similarity on normalized embeddings) |
| Scores in churn | Float in `[0.0, 1.0]` (probability). Logits are never exposed. |
| Null vs missing | Prefer explicit `null` over an absent field when semantically meaningful |
| Enum values | `snake_case`, lowercase (e.g. `"cache"`, `"ann"`, `"fallback"`) |
| Lists | Ordered, deterministic. **Tie-break by `item_id` ascending.** |
| Money / PII | Never in logs. Never in metrics labels. |
| Versioning | Every response carries `model_version` and `index_version` in `meta` |

**Rule:** if a value is derived from user input, it goes through an allowlist
before it can appear in a query, a cache key, a metric label, or a log field.

---

## 1. Data contracts — ingest

### 1.1 `EventEnvelope` (`POST /v1/events`)

Client-supplied. Idempotent by `event_id`.

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class ExperimentRef(BaseModel):
    """Identifies the experiment arm a user was actually served."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    variant: str = Field(min_length=1, max_length=64)


class EventEnvelope(BaseModel):
    """
    A single interaction event.

    Producer is the client; consumer is the event log table and the offline
    experiment analyzer. Add fields only via a new ADR; do not repurpose
    existing ones.
    """
    model_config = ConfigDict(extra="forbid")

    # Identity & idempotency ------------------------------------------------
    event_id: str = Field(
        description="Client-generated UUIDv4. Uniqueness enforced at ingest.",
        min_length=8, max_length=64,
    )
    event_ts: datetime = Field(
        description="When the event occurred, per the client. UTC.",
    )

    # Core fields -----------------------------------------------------------
    event_type: Literal["impression", "click", "conversion"]
    user_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)

    # Attribution (nullable: not all events come from a recommend response) -
    request_id: str | None = Field(default=None, max_length=64)
    experiment: ExperimentRef | None = None

    # Context (free-form but typed; no PII) ---------------------------------
    position: int | None = Field(
        default=None, ge=1,
        description="1-based rank of item_id in the served list, if applicable.",
    )
    value: float | None = Field(
        default=None,
        description="Monetary or weighted value for conversion events.",
    )
```

**Invariants**

- `event_type == "conversion"` → `value` MAY be set; otherwise MUST be `null`.
- `position` MUST be set for `click` and `impression`; MAY be `null` for `conversion`.
- `experiment` MUST be set if the event is attributable to a recommendation
  response that was served under an experiment; otherwise `null`.

**Server behavior**

- `event_ts` accepted within a bounded skew window (`[now - 7d, now + 5m]`).
  Outside → `422`.
- `user_id` is **hashed** with a per-environment salt before storage in any
  analytical table. The raw value is dropped after the write.

### 1.2 `RecommendRequest` / `RecommendResponse`

See § 2.1.

---

## 2. API contracts

All request and response bodies are JSON, `Content-Type: application/json`.
All timestamps are ISO 8601 UTC.

### 2.1 `POST /v1/recommend`

**Request**

```python
class RecommendFilters(BaseModel):
    """Allowlisted metadata filters. Unknown keys are rejected, not ignored."""
    model_config = ConfigDict(extra="forbid")

    category: str | None = None
    brand: str | None = None
    language: str | None = None
    # Allowlist expands via ADR; never accept arbitrary field names.


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    seed_item_ids: list[str] = Field(default_factory=list, max_length=50)
    k: int = Field(default=10, ge=1, le=100)
    filters: RecommendFilters | None = None
```

**Response**

```python
class RecommendItem(BaseModel):
    item_id: str
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)


class ExperimentAssignment(BaseModel):
    name: str
    variant: str


class RecommendMeta(BaseModel):
    source: Literal["cache", "ann", "fallback"]
    model_version: str
    index_version: str
    experiment: ExperimentAssignment | None = None


class RecommendResponse(BaseModel):
    request_id: str
    items: list[RecommendItem]
    meta: RecommendMeta
```

**Guarantees**

- `items` is sorted by `score` descending; ties broken by `item_id` ascending.
- `len(items) <= k`. May be smaller when filters are very selective.
- `rank` is contiguous starting at 1.
- `meta.source` reflects the actual data path taken for this request.
- `meta.experiment` echoes the assignment made by the assignment function;
  it is `null` when no experiment is active for this user.

### 2.2 `GET /v1/items/{item_id}/similar`

**Query parameters**

| Name | Type | Default | Notes |
|---|---|---|---|
| `k` | int | 10 | `1 <= k <= 100` |
| `filters` | CSV | — | Same allowlist as § 2.1, encoded as `key:value,key:value` |

**Response:** same `RecommendResponse` shape, with `meta.experiment` always
`null` (item-to-item does not participate in A/B assignment).

### 2.3 `POST /v1/events`

**Request:** `EventEnvelope` (§ 1.1). Accepts a single event.

**Response:** `202 Accepted`

```json
{ "accepted": 1, "duplicates": 0 }
```

Duplicates are counted, not rejected — retries are idempotent by `event_id`.

### 2.4 `POST /v1/churn/score`

**Request**

```python
class ChurnScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
```

**Response**

```python
class ChurnScoreResponse(BaseModel):
    request_id: str
    user_id: str
    score: float = Field(ge=0.0, le=1.0)
    model_version: str
    features_ts: datetime  # timestamp of the freshest feature used
```

### 2.5 Error envelope

Every non-2xx response uses this shape. No exceptions.

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Per-key rate limit exceeded.",
    "request_id": "7c1f9e2a",
    "details": { "retry_after_seconds": 12 }
  }
}
```

**`code` allowlist**

| HTTP | `code` | When |
|---|---|---|
| 400 | `bad_request` | Malformed request that is not a validation error |
| 401 | `unauthenticated` | Missing/invalid credential |
| 403 | `forbidden` | Valid credential, insufficient scope |
| 404 | `not_found` | Resource does not exist |
| 422 | `validation_error` | Pydantic validation failure |
| 429 | `rate_limited` | Per-key limit hit; `Retry-After` header set |
| 500 | `internal_error` | Unhandled; never leak stack traces |
| 503 | `unavailable` | Both ANN retrieval and fallback failed |

---

## 3. Config contract

See [`.env.example`](../.env.example) for values. Enums are validated at
startup; an invalid value aborts boot rather than falling back silently.

| Env var | Type | Allowed | Default |
|---|---|---|---|
| `APP_ENV` | enum | `dev`, `staging`, `prod` | `dev` |
| `LOG_LEVEL` | enum | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `LOG_FORMAT` | enum | `json`, `console` | `json` |
| `INDEX_BACKEND` | enum | `pgvector`, `faiss` | `pgvector` |
| `DEVICE` | enum | `cpu`, `cuda` | `cpu` |
| `METRICS_ENABLED` | bool | `true`, `false` | `true` |
| `HNSW_M` | int | `>= 2` | `16` |
| `HNSW_EF_CONSTRUCTION` | int | `>= HNSW_M` | `64` |
| `HNSW_EF_SEARCH` | int | `>= 1` | `100` |
| `ANN_TIMEOUT_MS` | int | `1..5000` | `120` |
| `CACHE_TTL_SECONDS` | int | `0..86400` (`0` disables) | `300` |
| `RATE_LIMIT_PER_MINUTE` | int | `>= 1` | `600` |

**Rule:** production startup requires `APP_ENV=prod` **and** non-empty
`API_KEYS` **and** non-default `JWT_SECRET`. Startup fails otherwise.

---

## 4. Telemetry contract

Observability is part of the contract, not an afterthought. Names below are
stable; renaming requires an ADR.

### 4.1 Metric names

Format: `recsys_<subsystem>_<name>_<unit>` (Prometheus conventions).

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `recsys_requests_total` | counter | `route`, `method`, `status`, `source` | `source` ∈ `cache`/`ann`/`fallback` |
| `recsys_request_duration_seconds` | histogram | `route`, `source`, `status` | Buckets include `0.2` s so the target is directly measurable |
| `recsys_cache_requests_total` | counter | `result` | `result` ∈ `hit`/`miss`/`bypass`/`error` |
| `recsys_fallback_total` | counter | `reason` | `reason` ∈ `db_down`/`ann_timeout`/`cold_start`/`index_missing` |
| `recsys_errors_total` | counter | `type` | `type` is a bounded allowlist, not a raw exception class |
| `recsys_embedding_drift_score` | gauge | `window` | Centroid cosine shift over recent queries vs reference |
| `recsys_active_index_info` | gauge | `index_version`, `model_version` | Always `1`; use `info` pattern for joins |
| `recsys_experiment_exposures_total` | counter | `experiment`, `variant` | Emitted when a variant is actually served |
| `recsys_rate_limit_degraded` | gauge | — | `1` when shared Redis limiter is unavailable |

**Cardinality guardrails**

- No label may take unbounded values (`user_id`, `request_id`, raw URLs).
- `error.type` is mapped to a fixed set: `auth`, `validation`, `retrieval`,
  `cache`, `upstream`, `internal`.
- `route` uses the **path template** (`/v1/items/{item_id}/similar`), never
  the resolved path.

### 4.2 Log fields (structlog JSON)

Every log line is a JSON object. Field names below are reserved.

| Field | Type | When |
|---|---|---|
| `ts` | string | Always. ISO 8601 UTC. |
| `level` | string | Always. `info`/`warning`/`error`. |
| `event` | string | Always. Short stable identifier, e.g. `recommend.served`. |
| `request_id` | string | Requests. Matches `RecommendResponse.request_id`. |
| `trace_id`, `span_id` | string | When tracing is active. |
| `route` | string | Requests. Path template. |
| `method` | string | Requests. |
| `status` | int | Requests. |
| `duration_ms` | number | Requests. |
| `source` | string | Recommend path taken. |
| `user_id_hash` | string | When user-scoped. **Never the raw `user_id`.** |
| `error.type`, `error.message` | string | On errors. `message` is sanitized. |

**Forbidden in logs:** raw `user_id`, `item_id` lists, request bodies,
`Authorization`/`X-API-Key` values, `JWT_SECRET`, connection strings.

### 4.3 Span names (OpenTelemetry)

| Span | Parent | Notes |
|---|---|---|
| `http.server.request` | — | Auto-instrumented by FastAPI middleware |
| `recsys.cache.get` | `http.server.request` | Attributes: `cache.key.kind`, `cache.result` |
| `recsys.ann.query` | `http.server.request` | Attributes: `index.version`, `filters.count`, `k` |
| `recsys.rerank` | `http.server.request` | Attributes: `rerank.weights`, `mmr.lambda` |
| `recsys.fallback` | `http.server.request` | Attributes: `fallback.reason` |
| `recsys.embed.encode` | — | Offline batch only; not on the hot path |

**Sampling:** 100% of errors, 10% of successes in `prod`, 100% in `dev`/`staging`.

---

## 5. Idempotency matrix

Filled here so that every write endpoint has an explicit answer to "what
happens on retry?".

| Operation | Key | Behavior on retry |
|---|---|---|
| `POST /v1/events` | `event_id` (client UUIDv4) | `INSERT ... ON CONFLICT (event_id) DO NOTHING`. Response counts duplicate. |
| Embed item | `sha256(model_version + content_hash)` | Skip if row exists with the same key. |
| Build index | `sha256(index_backend + params + catalog_snapshot + model_version)` | New `index_version` row; the live pointer does not move. |
| Promote index | `index_version` (target) | Idempotent: promoting the already-active version is a no-op. |
| A/B assignment | `sha256(f"{experiment_salt}:{user_id}") % 10_000` | Pure function; no state. |
| Cache write | `sha256(index_version + variant + filters + seed_items + k)` | Overwrite on same key; TTL bounds staleness. |
| Churn score | none (read-only) | Safe to retry. |

**Rule:** if an operation is not in this table, it must be a pure read.

---

## 6. Change management

- **Additive change** (new optional field, new enum value in a
  non-exhaustive position, new metric): PR + CHANGELOG entry. No ADR.
- **Behavioral change** (same schema, different semantics): ADR required.
- **Breaking change** (field removed, type changed, enum value removed):
  ADR + version bump of the API path (`/v2/...`). The previous path is kept
  until consumers migrate.
- **Contract drift between this document and code** is treated as a bug.
  The document wins; the code is fixed in the same PR.
