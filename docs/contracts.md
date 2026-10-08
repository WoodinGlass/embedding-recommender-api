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

## 1. Data contracts

### 1.1 `EventEnvelope` (`POST /v1/events`)
### 1.1 Event ingestion (`POST /v1/events`)

**Batch envelope.** The request body is a list of one to one hundred
`EventEnvelope` objects (ADR-0018). A single event is the list of one;
the schema is uniform.

```python
from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


class ExperimentRef(BaseModel):
    """Identifies the experiment arm a user was actually served."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    variant: str = Field(min_length=1, max_length=64)


class EventEnvelope(BaseModel):
    """
    A single interaction event. Producer is the client; consumer is the
    event log table and the offline experiment analyzer. Add fields only
    via a new ADR; do not repurpose existing ones.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity & idempotency
    event_id: str = Field(
        description=(
            "Client-generated unique id. Uniqueness enforced at ingest; "
            "a duplicate is counted, not rejected. If omitted, the server "
            "generates sha256(f'{request_id}:{index}') and returns it."
        ),
        min_length=8,
        max_length=64,
    )
    event_ts: datetime = Field(
        description=(
            "When the event occurred, per the client, UTC. Accepted within "
            "[now - 7 days, now + 5 minutes] (ADR-0018); outside → 422."
        ),
    )

    # Core fields
    event_type: Literal["impression", "click", "conversion"]
    user_id: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "Raw user id. The server HMAC-SHA256 hashes it with a versioned "
            "salt before storage; the raw value is discarded after the "
            "handler returns (ADR-0018)."
        ),
    )
    item_id: str = Field(min_length=1, max_length=128)

    # Attribution (nullable: not every event comes from a recommend response)
    request_id: str | None = Field(default=None, max_length=64)
    experiment: ExperimentRef | None = None

    # Context (free-form but typed; no PII)
    position: int | None = Field(
        default=None,
        ge=1,
        description="1-based rank of item_id in the served list, if applicable.",
    )
    value: float | None = Field(
        default=None,
        description="Monetary or weighted value for conversion events.",
    )


class EventBatch(BaseModel):
    """The `POST /v1/events` request body. One to one hundred events."""

    model_config = ConfigDict(extra="forbid")

    events: Annotated[list[EventEnvelope], Field(min_length=1, max_length=100)]


class EventAck(BaseModel):
    """The `202 Accepted` response body."""

    model_config = ConfigDict(extra="forbid")

    accepted: int = Field(ge=0)
    duplicates: int = Field(ge=0)
    rejected: int = Field(ge=0)
    generated_event_ids: list[str] | None = Field(
        default=None,
        description=(
            "Parallel to the request's events array; an entry for each event "
            "whose event_id the client omitted. null when the client "
            "supplied every id."
        ),
    )
```

**Invariants**

- `event_type == "conversion"` → `value` MAY be set; otherwise MUST be `null`.
- `position` MUST be set for `click` and `impression`; MAY be `null` for `conversion`.
- `experiment` MUST be set if the event is attributable to a recommendation
  response that was served under an experiment; otherwise `null`.

**Server behavior**

- **Skew window.** `event_ts` is accepted when it is within
  `[now - EVENT_TS_MAX_AGE_SECONDS, now + EVENT_TS_MAX_FUTURE_SECONDS]`
  (defaults: 7 days back, 5 minutes forward). Outside → `422` with the
  offending timestamp in `error.details`.
- **Idempotency.** The database has a unique constraint on `event_id`;
  the insert is `INSERT ... ON CONFLICT (event_id) DO NOTHING`. A
  duplicate is counted in `EventAck.duplicates`, not rejected.
- **PII hashing.** The handler computes
  `user_id_hash = HMAC_SHA256(USER_ID_HASH_SALT, user_id)` and stores the
  hash alongside `USER_ID_HASH_SALT_VERSION` (an integer). The raw
  `user_id` never appears in the database, the logs, the metrics, the
  traces, or an error message. `tests/unit/test_pii_redaction.py`
  asserts this by running the handler with a sentinel `user_id` and
  checking the captured output.
- **Retention.** Events older than `EVENT_RETENTION_DAYS` (default 365)
  are deleted by a scheduled job (M6). The policy is stated here even
  though the job lands later; the table's growth is a documented,
  bounded quantity.

**`POST /v1/events` response**

`202 Accepted` with an `EventAck` body. The status is `202` because the
events are accepted for recording, not because a downstream consumer has
processed them.

**Idempotency of the batch itself.** A retry of the same batch with the
same `event_id`s is idempotent at the row level: the second insert
conflicts and the `duplicates` count reflects it. A retry of the same
batch with server-generated ids is idempotent only when the retry uses
the same `request_id` (the generated ids derive from it). This is
documented so a client that does not supply `event_id` knows what makes
its retry safe.

### 1.2 `RecommendRequest` / `RecommendResponse`

See § 2.1.

### 1.3 Embedding artifacts (M1)

The embedding pipeline produces versioned artifacts on disk. Their names and
formats are contracts: anything that reads or writes them must conform. See
[`docs/embedding-pipeline.md`](embedding-pipeline.md) for the design and
[`docs/adr/0004-versioned-runs-with-current-pointer.md`](adr/0004-versioned-runs-with-current-pointer.md)
for why the layout is versioned rather than flat.

**`model_version`** — string, format `<label>+<sha8>`:

```
minilm-onnx-v1+a3f9e021
```

- `<label>` is a human-readable tag chosen by the operator.
- `<sha8>` is the first 8 hex characters of the SHA256 of the ONNX artifact
  bytes.
- The full 64-character SHA256 is recorded in the artifact manifest; only
  the 8-character prefix appears in `model_version` strings.
- Two artifacts with the same `<sha8>` are byte-identical. Two artifacts with
  different `<sha8>` are different models regardless of `<label>`.

**`catalog_snapshot`** — string, format `sha256:<16 hex>`:

```
sha256:9f1e2c8a3b5d7e4f
```

Computed as `sha256("\n".join(f"{item_id}:{content_hash}" for sorted items))`,
truncated to 16 hex characters. Properties:

- Row order in the source catalog does not affect the snapshot.
- A content change on any item (which changes its `content_hash`) changes
  the snapshot.
- Adding or removing items changes the snapshot.
- Two catalog exports with the same snapshot are semantically identical for
  embedding purposes.

**`content_hash` per item** — SHA256 of the preprocessed text (see
`docs/embedding-pipeline.md` § 4), truncated to 16 hex characters. Stored as
a column in the Parquet so incremental runs can decide "changed / unchanged"
without re-reading the raw catalog.

**Artifact layout — versioned runs with an atomic `current` pointer**:

```
artifacts/
├── onnx/
│   └── <model_slug>/
│       ├── model.onnx
│       ├── model.onnx.sha256
│       ├── tokenizer.json
│       └── config.json
└── embeddings/
    ├── runs/
    │   └── <run_id>/
    │       ├── embeddings.parquet      # the embedding table (the contract)
    │       ├── manifest.json           # metadata about the run
    │       └── state.json              # incremental sidecar (not a contract)
    ├── current                         # one-line text file: the active run_id
    └── .lock                           # fcntl.flock target
```

`model_slug` is the model name with `/` replaced by `__` and other
filesystem-hostile characters removed; e.g.
`sentence-transformers/all-MiniLM-L6-v2` →
`sentence-transformers__all-MiniLM-L6-v2`.

`run_id` is `<ISO8601-Z>__<model_version>`, e.g.
`2026-10-07T11-30-00Z__minilm-onnx-v1+a3f9e021`. Both components are also
recorded inside `manifest.json`, so parsing the run_id is never required.

**The `current` pointer** is a text file containing one line: the `run_id`
of the active run, terminated by a single newline. It is written last in
the commit sequence (see `docs/adr/0004-versioned-runs-with-current-pointer.md`).
Consumers read `artifacts/embeddings/current` and then
`runs/<run_id>/manifest.json`; they never reconstruct a run_id or scan the
`runs/` directory.

**The manifest** is the source of truth for what a run contains:

```json
{
  "schema_version": 1,
  "created_at": "2026-10-07T11:30:00Z",
  "run_id": "2026-10-07T11-30-00Z__minilm-onnx-v1+a3f9e021",
  "mode": "incremental",
  "model_version": "minilm-onnx-v1+a3f9e021",
  "preprocessing_version": "v1",
  "config_hash": "sha256:...",
  "catalog_snapshot": "sha256:...",
  "onnx_artifact_sha256": "e9c6...",
  "parquet": {
    "path": "runs/2026-10-07T11-30-00Z__minilm-onnx-v1+a3f9e021/embeddings.parquet",
    "sha256": "...",
    "rows": 200,
    "encoded_rows": 0,
    "dim": 384,
    "dtype": "float32"
  },
  "state": {
    "path": "runs/2026-10-07T11-30-00Z__minilm-onnx-v1+a3f9e021/state.json",
    "sha256": "..."
  },
  "environment": {
    "python_version": "3.11.16",
    "onnxruntime_version": "1.19.0",
    "numpy_version": "2.1.0",
    "pyarrow_version": "17.0.0"
  }
}
```

Paths inside `manifest.json` are **relative to `artifacts/embeddings/`**, so
the directory can be relocated without rewriting the manifest.

**`config_hash`** is a SHA256 over a canonical JSON object containing every
parameter that changes the embedding content:

```
{
  "preprocessing_version": "v1",
  "onnx_artifact_sha256": "e9c6...",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "max_seq_length": 256,
  "embedding_dim": 384,
  "pooling": "mean",
  "normalize": true
}
```

Thread count and batch size are **excluded**: they affect bit-level noise
within the tolerance band defined in § 5 of `docs/embedding-pipeline.md`, but
not the embedding's meaning. Including them would force a full re-encode on
every CI run.

`state.json` is not part of the contract — see `docs/embedding-pipeline.md`
§ 7.1. Deleting it only costs performance.

**Determinism contract** — three tiers. Full rationale in
`docs/embedding-pipeline.md` § 5.

| Tier | Assertion | Runs where |
|---|---|---|
| Strict | SHA256 of `*.parquet` matches between two consecutive runs in the same environment | CI only |
| Semantic | Top-k (k=10) neighbours for a fixed probe set are identical — same item IDs, same order, ties broken by `item_id` ascending | CI and local |
| Tolerance | Per-row cosine similarity ≥ 0.9999 between two runs | Local only |

### 1.4 Index registry and index identity (M2)

M2 introduces a persistent index registry and an `index_version` identifier.
Both are contracts: anything that reads or writes the `index_registry` table,
or that puts an `index_version` in a response, must conform. Full rationale
in `docs/adr/0006-pgvector-schema.md` and
`docs/adr/0007-index-version-identity.md`; the shape below is the contract.

**`index_version` format** — string, `idx-<8 hex>` (extended to 12 hex on
collision):

```
idx-a3f9e021
```

Computed as the first 8 hex characters of the SHA256 over a canonical JSON
object containing exactly these fields:

```
{
  "model_version":         "minilm-onnx-v1+a3f9e021",
  "catalog_snapshot":      "sha256:9f1e2c8a3b5d7e4f",
  "preprocessing_version": "v1",
  "metric":                "cosine",
  "hnsw_m":                16,
  "hnsw_ef_construction":  64,
  "pgvector_version":      "0.8.0"
}
```

- Canonicalization: `sort_keys=True`, `separators=(",", ":")`, UTF-8, no
  trailing newline.
- **Excluded by design:** `hnsw_ef_search` (a query-time knob; changing it
  does not require a rebuild) and `golden_set_version` (evaluation
  metadata; changing the golden set does not require a rebuild).
- **Collision handling:** on a collision where the existing row's hash
  inputs differ, extend to 12 hex. On a collision where they match, treat
  the build as a duplicate and do not create a row.
- **Capture time:** `pgvector_version` is captured at build time from
  `SELECT extversion FROM pg_extension WHERE extname = 'vector'` and stored
  in `index_registry`. It is not re-queried at runtime.

**`index_registry` schema** — the source of truth for what an index
contains and how it behaves:

| Column | Type | Notes |
|---|---|---|
| `index_version` | `TEXT` (PK) | Format above |
| `model_version` | `TEXT` | Matches M1 `model_version` |
| `catalog_snapshot` | `TEXT` | Matches M1 `catalog_snapshot` |
| `preprocessing_version` | `TEXT` | Matches M1 `preprocessing_version` |
| `metric` | `TEXT` | `cosine` \| `l2` \| `ip` |
| `hnsw_m` | `INTEGER` | Build parameter |
| `hnsw_ef_construction` | `INTEGER` | Build parameter |
| `hnsw_ef_search` | `INTEGER` | Default query-time knob |
| `pgvector_version` | `TEXT` | Captured at build time |
| `golden_set_version` | `TEXT` | Golden set the thresholds apply to |
| `row_count` | `INTEGER` | Number of embeddings in this index |
| `status` | `TEXT` | `building` \| `active` \| `retired` |
| `created_at` | `TIMESTAMPTZ` | |
| `activated_at` | `TIMESTAMPTZ` (nullable) | Set when status becomes `active` |
| `retired_at` | `TIMESTAMPTZ` (nullable) | Set when status becomes `retired` |

**State transitions** are constrained: `building` → `active` → `retired`.
An index in `active` is unique, enforced by a partial unique index on
`status = 'active'`. Nothing else is a valid transition; a caller that
attempts one receives an error from the database, not from application
code.

**Where `index_version` appears**:

- `embedding.index_version` — foreign key; every embedding row belongs to
  exactly one index.
- `RecommendResponse.meta.index_version` — § 2.1.
- `recsys_active_index_info{index_version=...}` — § 4.1 (metric label).

**`catalog_snapshot` and `preprocessing_version` are not recomputed** at
query time. They describe the artifact an index was built from; the
registry row is authoritative.

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

**Filter allowlist — `FILTER_FIELDS`.** The set of filter fields is
`{"category", "brand", "language"}`, defined once in
`src/recsys/retrieval/filters.py` as `FILTER_FIELDS` and referenced by both
the API schemas and the retrieval layer. It is not duplicated in the
retrieval code. Adding a field requires a change to this section of
`contracts.md` and to `FILTER_FIELDS` in the same PR. Filter values are
always passed to the database as parameters, never interpolated into SQL;
a value outside the allowlist is rejected by the API at `422` before it
reaches retrieval.


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
    source: Literal[
        "cache",
        "ann",
        "fallback_ann",
        "fallback_cached",
        "none",
    ]
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
  The five values are:

  | Value | Meaning |
  |---|---|
  | `cache` | Served from Redis (ADR-0015). |
  | `ann` | Served from the ANN index, re-ranked (ADR-0016). |
  | `fallback_ann` | The ANN path failed or timed out; a popular list read from the database was returned (ADR-0020 § tier 3). |
  | `fallback_cached` | The database was unreachable; the same popular list, held in process memory, was returned (ADR-0020 § tier 4). |
  | `none` | Every tier failed; the response is `503` (ADR-0020 § tier 5). |

  A client that switches on `source` should treat `fallback_ann` and
  `fallback_cached` as "the recommendation is popular, not personalized".
  The scores in a fallback response are a placeholder in `(0, 1]`;
  they are not comparable to an ANN score.

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
| 429 | `rate_limited` | Per-key limit hit; `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers set |
| 500 | `internal_error` | Unhandled; never leak stack traces |
| 503 | `unavailable` | Both ANN retrieval and fallback failed |

---

### 2.6 Health and readiness endpoints

Two endpoints answer two different questions (ADR-0019). A third is kept
as an alias for backwards compatibility.

#### `GET /livez` and `GET /healthz`

Liveness. Answers "is this process alive enough to keep running" and
touches no dependency. `200` with:

```json
{ "status": "alive" }
```

`/healthz` is an alias: identical body, identical behavior. The M0
scaffold and the CI Docker smoke test call `/healthz`; `/livez` is the
name the Kubernetes liveness probe uses.

#### `GET /readyz`

Readiness. Answers "should this instance receive traffic right now".
Returns a three-state `status` and a structured `checks` object.

```json
{
  "status": "ready",
  "checks": {
    "db":    { "ok": true,  "latency_ms": 2 },
    "index": { "ok": true,  "index_version": "idx-a3f9e021", "row_count": 200 },
    "redis": { "ok": true,  "required": false, "latency_ms": 1 }
  }
}
```

| `status` | HTTP | Meaning |
|---|---|---|
| `ready` | 200 | Every required check is `ok`. |
| `degraded` | 200 | Every required check is `ok`; at least one optional check is not. The instance can serve, more slowly or with less functionality. |
| `not_ready` | 503 | At least one required check is not `ok`. The instance should be removed from rotation. |

`db` and `index` are required. `redis` is optional; its `checks.redis.required`
field is `false`, so a caller does not have to read this document to know
which check can be ignored.

**Failed checks carry an `error` field:**

```json
{
  "status": "degraded",
  "checks": {
    "db":    { "ok": true,  "latency_ms": 2 },
    "index": { "ok": true,  "index_version": "idx-a3f9e021", "row_count": 200 },
    "redis": { "ok": false, "required": false, "error": "timeout" }
  }
}
```

The `error` value is one of a closed set: `timeout`, `connection_refused`,
`auth_failed`, `not_configured`, `unknown`.

**Timeouts and caching.** Each check has its own hard timeout (500 ms
for `db` and `index`, 100 ms for `redis`); a check that exceeds it is
recorded as `ok: false` with `error: "timeout"`. The full set of checks
runs at most once every `READYZ_CACHE_SECONDS` (default 5); a second
call within the window returns the cached result without touching a
dependency. The cache is per process.

**HTTP status for `degraded` is 200.** A Redis hiccup must not drain a
healthy instance; the body carries the degradation, the status code
carries "can this instance serve" (yes).

## 3. Config contract

See [`.env.example`](../.env.example) for values. Enums are validated at
startup; an invalid value aborts boot rather than falling back silently.
The five configuration categories (boot, hot, static, runtime,
evaluation) and the rules for which value belongs where are in
`docs/adr/0022-config-management.md`.

### 3.1 Boot config (environment variables)

| Env var | Type | Allowed | Default |
|---|---|---|---|
| **Application** | | | |
| `APP_ENV` | enum | `dev`, `staging`, `prod` | `dev` |
| `LOG_LEVEL` | enum | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `LOG_FORMAT` | enum | `json`, `console` | `json` |
| `METRICS_ENABLED` | bool | `true`, `false` | `true` |
| **Auth** (ADR-0013) | | | |
| `API_KEYS` | semicolon-list of hashes | non-empty in prod | `dev-key-1` |
| `API_KEYS_ADMIN` | comma-list of hashes | empty allowed | empty |
| `JWT_SECRET` | string | ≥ 32 chars, not the sentinel, in prod | sentinel |
| `JWT_ALGORITHM` | enum | `HS256`, `RS256` | `HS256` |
| `JWT_MAX_AGE_SECONDS` | int | `> 0` | `86400` |
| **Rate limit** (ADR-0014) | | | |
| `RATE_LIMIT_PER_MINUTE` | int | `>= 1` | `600` |
| `RATE_LIMIT_IP_PER_MINUTE` | int | `>= 1` | `5000` |
| `INSTANCE_COUNT` | int | `>= 1` | `1` |
| `TRUSTED_PROXY_COUNT` | int | `0..10` | `0` |
| **Breaker** (ADR-0015) | | | |
| `REDIS_BREAKER_FAILURE_THRESHOLD` | int | `>= 1` | `5` |
| `REDIS_BREAKER_OPEN_SECONDS` | float | `> 0` | `5` |
| `REDIS_BREAKER_OPEN_MAX_SECONDS` | float | `>= OPEN_SECONDS` | `60` |
| `REDIS_BREAKER_TIMEOUT_SECONDS` | float | `> 0` | `0.1` |
| `DB_BREAKER_FAILURE_THRESHOLD` | int | `>= 1` | `5` |
| `DB_BREAKER_OPEN_SECONDS` | float | `> 0` | `2` |
| `DB_BREAKER_OPEN_MAX_SECONDS` | float | `>= OPEN_SECONDS` | `30` |
| `DB_BREAKER_TIMEOUT_SECONDS` | float | `> 0` | `2.0` |
| **Event ingestion** (ADR-0018) | | | |
| `EVENT_TS_MAX_AGE_SECONDS` | int | `> 0` | `604800` (7 days) |
| `EVENT_TS_MAX_FUTURE_SECONDS` | int | `>= 0` | `300` (5 min) |
| `USER_ID_HASH_SALT` | string | non-empty in prod | sentinel |
| `USER_ID_HASH_SALT_VERSION` | int | `>= 1` in prod | `0` |
| `EVENT_RETENTION_DAYS` | int | `> 0` | `365` |
| **Readiness** (ADR-0019) | | | |
| `READYZ_CACHE_SECONDS` | int | `0..60` (`0` disables) | `5` |
| `READYZ_DB_TIMEOUT_SECONDS` | float | `> 0` | `0.5` |
| `READYZ_INDEX_TIMEOUT_SECONDS` | float | `> 0` | `0.5` |
| `READYZ_REDIS_TIMEOUT_SECONDS` | float | `> 0` | `0.1` |
| **Observability** (ADR-0021) | | | |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | string | — | `http://otel-collector:4317` |
| `OTEL_SERVICE_NAME` | string | — | `recsys-api` |
| `OTEL_TRACES_SAMPLER_ARG` | float | `0.0..1.0` | `0.1` in prod, `1.0` otherwise |
| **Config pollers** (ADR-0022) | | | |
| `HOT_CONFIG_POLL_SECONDS` | int | `>= 1` | `5` |
| `ACTIVE_INDEX_POLL_SECONDS` | int | `>= 1` | `10` |
| **Embeddings / retrieval** (M1–M2) | | | |
| `DATABASE_URL` | string | parseable URL in prod | local |
| `REDIS_URL` | string | parseable URL in prod | local |
| `EMBEDDING_MODEL` | string | — | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_ONNX_PATH` | string | must match `export_onnx.py` output | `artifacts/onnx/sentence-transformers__all-MiniLM-L6-v2` |
| `EMBEDDING_BATCH_SIZE` | int | `>= 1` | `64` |
| `INDEX_BACKEND` | enum | `pgvector`, `faiss` | `pgvector` |
| `DEVICE` | enum | `cpu`, `cuda` | `cpu` |
| `HNSW_M` | int | `>= 2` | `16` |
| `HNSW_EF_CONSTRUCTION` | int | `>= HNSW_M` | `64` |
| `HNSW_EF_SEARCH` | int | `>= 1` | `100` |
| `ANN_TIMEOUT_MS` | int | `1..5000` | `120` |
| `CACHE_TTL_SECONDS` | int | `0..86400` (`0` disables) | `300` |
| **Deployment** (ADR-0023) | | | |
| `PRE_STOP_DELAY_SECONDS` | int | `>= 0` | `5` |
| **SLO** (ADR-0024) | | | |
| `LATENCY_SLO_MS` | int | `> 0` | `200` |

**Prod guard.** Production startup requires `APP_ENV=prod` **and**:

- `API_KEYS` non-empty
- `JWT_SECRET` not the sentinel value and ≥ 32 characters
- `USER_ID_HASH_SALT` set (not the sentinel) and
  `USER_ID_HASH_SALT_VERSION >= 1`
- `DATABASE_URL` and `REDIS_URL` present and parseable

`API_KEYS_ADMIN` may be empty (a deployment with no admin operations is
legal). Startup fails on any of the checks above; the failure is at
process start, not on the first request that needs the value.

### 3.2 Hot config (`config/hot.yaml`)

Read at startup and re-read on every change within
`HOT_CONFIG_POLL_SECONDS` (ADR-0022). A malformed change is logged and
the previous value stays in effect.

| Field | Type | Allowed | Default |
|---|---|---|---|
| `rate_limit.recommend_per_minute` | int | `>= 1` | `600` |
| `rate_limit.events_per_minute` | int | `>= 1` | `6000` |
| `rate_limit.admin_per_minute` | int | `>= 1` | `60` |
| `rerank.w_sim` | float | `[0, 1]` | `0.7` |
| `rerank.w_pop` | float | `[0, 1]` | `0.2` |
| `rerank.w_rec` | float | `[0, 1]` | `0.1` |
| `rerank.mmr_lambda` | float | `[0, 1]` | `0.7` |
| `rerank.mmr_window` | int | `>= k` | `50` |
| `rerank.mmr_min_k` | int | `>= 1` | `10` |
| `rerank.candidate_multiplier` | int | `>= 1` | `4` |
| `rerank.recency_half_life_days` | float | `> 0` | `90` |
| `cache.recommend_ttl_seconds` | int | `0..86400` | `300` |
| `cache.similar_ttl_seconds` | int | `0..86400` | `600` |
| `cache.negative_ttl_seconds` | int | `0..3600` | `30` |

The re-ranker weights are validated to sum to `> 0` (not exactly `1`);
a weight of `0` disables that signal.

### 3.3 Static config (`experiments.yaml`)

Declared at the repository root and read once at startup (ADR-0017).
The schema is in that ADR; the file is validated before the app serves
a request. `EXPERIMENT_DISABLED=1` (an environment variable) treats
every experiment as stopped for this process.

### 3.4 Runtime state and evaluation config

The active index is a row in `index_registry` (ADR-0006) read with a
`ACTIVE_INDEX_POLL_SECONDS` cache (ADR-0022). The evaluation thresholds
(`evaluation/thresholds.yaml`, ADR-0010) are read by `scripts/eval.py`
and by CI; no module under `src/recsys/api/` imports them.

### 3.5 Secrets

Secrets — `JWT_SECRET`, `API_KEYS`, `API_KEYS_ADMIN`,
`USER_ID_HASH_SALT`, and the password in `DATABASE_URL` — are environment
variables only. They are never in a committed file, a hot config, a
database row, or a log line (ADR-0021 § Logs, ADR-0022 § Secrets).


## 4. Telemetry contract

Observability is part of the contract, not an afterthought. Names below are
stable; renaming requires an ADR.

### 4.1 Metric names

Format: `recsys_<subsystem>_<name>_<unit>` (Prometheus conventions). The
table below is the complete registry; a metric emitted by the code but
not listed here is a contract violation caught by
`tests/unit/test_metric_contract.py`.

| Metric | Type | Labels | Notes |
|---|---|---|---|
| `recsys_requests_total` | counter | `route`, `method`, `status`, `source` | `source` ∈ `cache`/`ann`/`fallback_ann`/`fallback_cached` (ADR-0020 extends the M0 set) |
| `recsys_request_duration_seconds` | histogram | `route`, `source`, `status` | Buckets include `0.2` s so the latency SLO (ADR-0024) is directly measurable |
| `recsys_cache_requests_total` | counter | `result` | `result` ∈ `hit`/`miss`/`bypass`/`error` (ADR-0015) |
| `recsys_cache_negative_hits_total` | counter | — | Hits on a negative entry (ADR-0015) |
| `recsys_cache_write_errors_total` | counter | `type` | A write that failed after a successful read (ADR-0015) |
| `recsys_fallback_total` | counter | `tier` | `tier` ∈ `fallback_ann`/`fallback_cached` (ADR-0020; the M0 `reason` label is renamed) |
| `recsys_errors_total` | counter | `type` | `type` ∈ `auth`/`validation`/`retrieval`/`cache`/`upstream`/`internal` |
| `recsys_rerank_duration_seconds` | histogram | `arm` | Time spent in the re-ranker, by experiment arm (ADR-0016) |
| `recsys_rerank_failures_total` | counter | `step`, `type` | `step` ∈ `normalize`/`blend`/`mmr` (ADR-0016) |
| `recsys_rerank_skipped_total` | counter | `reason` | `reason` ∈ `no_vectors`/`k_below_threshold` (ADR-0016) |
| `recsys_rate_limit_hits_total` | counter | `class_name`, `result` | `result` ∈ `allowed`/`limited` (ADR-0014) |
| `recsys_rate_limit_remaining` | histogram | `class_name` | Tokens left at decision time (ADR-0014) |
| `recsys_rate_limit_degraded` | gauge | — | `1` when the per-instance fallback limiter is active (ADR-0014) |
| `recsys_rate_limit_lua_errors_total` | counter | `type` | Lua script failures by category (ADR-0014) |
| `recsys_circuit_breaker_state` | gauge | `name` | `name` ∈ `redis`/`postgres`; `0` closed, `1` half-open, `2` open (ADR-0015) |
| `recsys_circuit_breaker_trips_total` | counter | `name`, `reason` | `reason` ∈ `threshold`/`probe_failed` (ADR-0015) |
| `recsys_circuit_breaker_state_changes_total` | counter | `name`, `from`, `to` | Breaker transitions; `from`/`to` ∈ `closed`/`half_open`/`open` (ADR-0015) |
| `recsys_experiment_exposures_total` | counter | `experiment`, `variant` | Both from `experiments.yaml`; emitted when a variant is served (ADR-0017) |
| `recsys_experiment_exposure_errors_total` | counter | `type` | A failed exposure write (ADR-0017) |
| `recsys_readyz_status` | gauge | `status` | `status` ∈ `ready`/`degraded`/`not_ready` (ADR-0019) |
| `recsys_readyz_check_duration_seconds` | histogram | `check` | `check` ∈ `db`/`index`/`redis` (ADR-0019) |
| `recsys_active_index_info` | gauge | `index_version`, `model_version` | Always `1`; `info` pattern for joins (M2) |
| `recsys_embedding_drift_score` | gauge | `window` | Centroid cosine shift; M4 (ADR-0021 defers) |

**Cardinality guardrails** (ADR-0021 § Metrics)

- No label may take unbounded values (`user_id`, `item_id`, `request_id`,
  `trace_id`, `event_id`, raw URLs, a resolved path).
- A label value is drawn from a set known at build time: an enum, a
  route template, a config-defined class, an experiment name from
  `experiments.yaml`.
- `error.type` and the other `type` labels are mapped to closed sets;
  the code contains the mapping function, not `type(exc).__name__`.
- `route` uses the path template (`/v1/items/{item_id}/similar`), never
  the resolved path.
- The label set on a metric is fixed for the metric's lifetime. Adding
  a label is a contract change (this section is updated in the same PR).


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

### 4.4 Retrieval log events (M2)

The retrieval layer emits structured log lines through
:mod:`recsys.monitoring.logging` (M0). The `event` names below are stable;
renaming requires an ADR. All are JSON, all carry the standard reserved
fields from § 4.2 (`ts`, `level`, `event`), and none contain PII or raw
query vectors.

| Event | Level | When | Extra fields |
|---|---|---|---|
| `retrieval.pgvector.version` | INFO | Once per process, at first use | `detected` (string, e.g. `"0.8.0"`) |
| `retrieval.pgvector.old_version` | WARNING | Once per process, when detected version < 0.8 | `detected`, `required_for_iterative_scan`, `fallback` |
| `retrieval.pgvector.fallback` | WARNING | Per filtered query on pgvector < 0.8 | `reason`, `detected`, `requested_k`, `effective_k` |
| `index.build.done` | INFO | After a build commits | `index_version`, `model_version`, `catalog_snapshot`, `row_count`, `duration_ms` |
| `index.build.duplicate` | INFO | When a build is a no-op (identical inputs) | `index_version` |
| `index.build.error` | ERROR | On a build failure | `index_version` (if allocated), `error.type`, `error.message` |
| `index.promote.done` | INFO | After a promote commits | `index_version`, `previous_index_version`, `duration_ms` |
| `index.promote.error` | ERROR | On a promote failure | `index_version`, `error.type`, `error.message` |
| `index.rollback.done` | INFO | After a rollback commits | `index_version`, `previous_index_version` |

**`retrieval.pgvector.fallback` is not silenced in production.** It is
the signal that an environment is running on an older extension and
serving filtered queries with a heuristic over-fetch. Suppressing it would
remove the only per-invocation evidence of the degraded path.

**Log volume.** `retrieval.pgvector.version` and
`retrieval.pgvector.old_version` fire exactly once per process. The other
retrieval events fire on user-driven actions (a fallback query, a build, a
promote). No per-request log line is emitted from the retrieval layer
itself; the access log (§ 4.2, `http.request`) covers request-level
tracing, and it carries the `source` label from § 4.1.

### 4.5 API log events (M3)

The events below are emitted by the API and its middleware. They follow
the same rules as § 4.4 (`event` naming convention, `schema_version=1`,
reserved fields, no raw credential, no raw `user_id`).

| Event | Level | When | Extra fields |
|---|---|---|---|
| `http.request` | INFO | Every request, on the response path | `route`, `method`, `status`, `duration_ms`, `source` |
| `auth.ok` | DEBUG | A credential validated | `principal.kind` (`api_key` / `jwt`), `scopes.count` |
| `auth.failed` | WARNING | A credential was rejected | `reason` ∈ `missing`/`invalid`/`expired`/`iat_too_old`/`unknown_kid`/`no_scope`; never the credential |
| `auth.validator_unavailable` | ERROR | The validator could not run (fail-closed path, ADR-0013) | `error.type` |
| `ratelimit.limited` | INFO | A request was refused | `class`, `credential_hash_prefix`, `retry_after_seconds` |
| `ratelimit.degraded` | WARNING | Once per process when the fallback limiter activates, once when it deactivates | `instance_count`, `effective_limit_per_minute` |
| `cache.hit` | DEBUG | A cache lookup hit | `cache.key.kind` |
| `cache.miss` | DEBUG | A cache lookup missed | `cache.key.kind` |
| `cache.bypass` | WARNING | The breaker is open; the lookup was skipped | `name` (`redis`), `reason` |
| `cache.error` | WARNING | A cache read or write raised | `operation` (`get`/`set`), `error.type` |
| `retrieval.fallback.ann` | WARNING | Tier 3 was reached | `reason` ∈ `timeout`/`error`, `ann_timeout_ms`, `latency_ms` |
| `retrieval.fallback.cached` | WARNING | Tier 4 was reached | `reason` ∈ `db_breaker_open`/`db_error`, `cache_age_seconds`, `row_count` |
| `retrieval.fallback.none` | ERROR | Tier 5; the response is `503` | the reasons every tier failed |
| `rerank.mmr.skipped` | DEBUG | MMR was not applied | `reason` ∈ `no_vectors`/`k_below_threshold` |
| `rerank.signal.missing` | WARNING | A re-ranker signal provider raised; treated as neutral | `signal` (`popularity`/`recency`) |
| `rerank.failed` | WARNING | A re-ranker step raised; the retrieval list is returned | `step` ∈ `normalize`/`blend`/`mmr`, `error.type` |
| `experiment.assigned` | DEBUG | A variant was assigned | `experiment`, `variant`, `bucket` |
| `experiment.exposure.written` | DEBUG | The exposure row was written | `experiment`, `variant` |
| `experiment.exposure.failed` | WARNING | The exposure write raised | `experiment`, `error.type` |
| `experiment.disabled` | WARNING | `EXPERIMENT_DISABLED=1` and a request would have been assigned | `experiment` |
| `events.ingested` | INFO | A batch was written | `accepted`, `duplicates`, `rejected`, `batch_size` |
| `events.rejected` | WARNING | A batch item was rejected by validation | `index`, `reason` |
| `readyz.check.failed` | WARNING | A readiness check returned not-ok | `check` (`db`/`index`/`redis`), `error` |
| `readyz.status.changed` | INFO | The three-state status changed between checks | `from`, `to` |

**`auth.failed` is the only auth log line at WARNING.** A `auth.ok`
line is DEBUG so a healthy request does not add an INFO line per
attempt; the access log carries the request-level detail.

**The credential never appears.** `credential_hash_prefix` is the first
8 hex characters of the BLAKE2b hash the rate limiter uses (ADR-0014),
not the credential.

## 5. Idempotency matrix

Filled here so that every write endpoint has an explicit answer to "what
happens on retry?".

| Operation | Key | Behavior on retry |
|---|---|---|
| `POST /v1/events` | `event_id` (client-supplied, or server-generated `sha256(f"{request_id}:{index}")`) | `INSERT ... ON CONFLICT (event_id) DO NOTHING`. The response counts the duplicate in `EventAck.duplicates`. |
| Exposure write | `sha256(f"exposure:{experiment}:{request_id}")` | Same table as `POST /v1/events`; a duplicate is ignored. The exposure is written once per served variant per request (ADR-0017). |
| `POST /v1/recommend` | None (read-only). The cache write is best-effort. | Safe to retry. The response's `request_id` differs; the cache write is idempotent under the same key (ADR-0015). |
| `GET /v1/items/{item_id}/similar` | None (read-only). | Safe to retry. |
| A/B assignment | `sha256(f"{APP_ENV}:{experiment_salt}:{user_id}") % 10_000` | Pure function; no state. The same `(user_id, experiment, environment)` produces the same bucket (ADR-0017). |
| Cache write | `cache:v1:{class}:{blake2b-16(request)}[:{variant}]` | Overwrite on the same key. The TTL and jitter bound staleness (ADR-0015). |
| Embed item | `sha256(model_version + content_hash)` | Skip if a row exists with the same key (M1). |
| Build index | `sha256(index_backend + params + catalog_snapshot + model_version + preprocessing_version + metric + pgvector_version)` | A new `index_version` row; the live pointer does not move (M2, ADR-0007). |
| Promote index | `index_version` (target) | Idempotent: promoting the already-active version is a no-op (M2). |
| Rollback index | The most recently retired `index_version` | Idempotent: rolling back when there is nothing retired raises `PromoteError` with a clear message (M2). |
| Churn score | None (read-only) | Safe to retry. *(M7)* |

**Rule:** if an operation is not in this table, it must be a pure read.
A write that is not here is a contract violation caught by review.

**Why `POST /v1/recommend` is a read, not a write.** The request does
not modify the catalog, the index, or the experiment state. It writes to
the cache and to the exposure log, but both writes are keyed by a
request-derived value and are idempotent under that key. From the
client's perspective, the request is safe to retry: the same inputs
produce the same result, and the duplicate writes are bounded.


## 6. Change management

- **Additive change** (new optional field, new enum value in a
  non-exhaustive position, new metric): PR + CHANGELOG entry. No ADR.
- **Behavioral change** (same schema, different semantics): ADR required.
- **Breaking change** (field removed, type changed, enum value removed):
  ADR + version bump of the API path (`/v2/...`). The previous path is kept
  until consumers migrate.
- **Contract drift between this document and code** is treated as a bug.
  The document wins; the code is fixed in the same PR.
