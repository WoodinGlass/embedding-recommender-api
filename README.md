# embedding-recommender-api

[![CI](https://github.com/WoodinGlass/embedding-recommender-api/actions/workflows/ci.yml/badge.svg)](https://github.com/WoodinGlass/embedding-recommender-api/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Embedding-based recommendation service with low-latency ANN retrieval (target p95 < 200 ms), re-ranking, statistically valid A/B testing, monitoring, and automated deployment (Docker + CI/CD). It ships with offline evaluation, model/index versioning, fallbacks, and a churn-risk extension.

> **Status:** in development. M0 (Foundation), M1 (Embedding pipeline), M2 (Retrieval and offline evaluation), and M3 (Production API) are complete; see [Milestones](#milestones). Performance figures for the served path are targets until the M4 benchmark is published.

## What this is / is not

**This is**

- A production-style recommendation service: embed a catalog, retrieve neighbors with ANN search and metadata filters, re-rank, and serve the result through FastAPI.
- A reference for the full ML-serving lifecycle: versioned models and indexes, offline evaluation as a CI gate, A/B experimentation, observability, graceful degradation, and zero-downtime index swaps.
- A single-node system sized for catalogs of 100k+ items.
- A foundation reused for a second use case: a churn-risk scoring endpoint on the same infrastructure.

**This is not**

- A chatbot or RAG system. Embeddings are used for item recommendation, not text generation.
- A vector database. pgvector and FAISS sit behind a small retrieval interface; the project does not try to replace Milvus, Qdrant, or Pinecone.
- A model-training project. It uses pre-trained sentence-transformers models without fine-tuning. The churn model is the only model trained here.
- A web-scale system. Sharding, multi-region replication, and catalogs of 10M+ items are out of scope.
- A turnkey product. There is no admin UI, multi-tenancy, or billing.

## Features

1. **Embedding pipeline** — batch and incremental catalog embedding with versioned models and indexes, atomic commit, and a three-tier determinism contract.
2. **Similarity search with metadata filters** — pgvector HNSW with a bounded, parameterized filter strategy; target p95 < 200 ms on 100k+ items.
3. **Offline evaluation** — Recall@k, NDCG@k, MRR, and ANN fidelity against a versioned golden set, with baselines and thresholds that fail the build.
4. **Re-ranking** — popularity, recency, and diversity (MMR) composed over the retrieval candidate list. Configurable weights, hot-reloadable via `config/hot.yaml` (ADR-0016, ADR-0022).
5. **Five-tier fallback chain** — `cache` → `ann` → `fallback_ann` (DB snapshot) → `fallback_cached` (in-memory) → `none` (503). Client errors never enter the chain (ADR-0020).
6. **A/B testing** — deterministic assignment (hashed user ID), SRM check, sample-size calculation, significance tests, and guardrail metrics. Assignment and exposure logging land in M3.6; the statistical analysis in M5.
7. **API hardening** — authentication (API key + JWT), Redis-backed rate limiting, input validation, and readiness/liveness endpoints (ADR-0013, ADR-0014, ADR-0019).
8. **Observability** — latency histograms, cache hit rate, fallback rate, error rate, and embedding drift, with alerts and Grafana dashboards. Metrics and logs are wired in M3; dashboards, alerts, and drift land in M4 (ADR-0021).
9. **Churn-risk extension** — separate endpoint, evaluated with AUC and calibration. *(M7)*
10. **Zero-downtime index swap** — blue/green indexes with instant rollback.

## Architecture

```mermaid
flowchart TB
  subgraph OFFLINE["Offline / batch (plain Python CLI; orchestration lands in M6)"]
    CAT[("Catalog")] --> EMB["Embed job<br/>ONNX encoder"]
    EMB --> RUNS[("Versioned runs<br/>artifacts/embeddings/ + current")]
    RUNS --> BLD["Build HNSW index<br/>(new index_version)"]
    BLD --> IDXREG[("index_registry<br/>+ embedding table<br/>(PostgreSQL)")]
    IDXREG --> PROM["Promote<br/>(blue/green pointer)"]
    IDXREG --> EVAL["Offline eval gate<br/>Recall@k, NDCG, MRR, ANN fidelity"]
    GS[("Golden set v1")] --> EVAL
    EVAL --> GATE{"Below threshold<br/>or absolute floor?"}
  end

  subgraph ONLINE["Online serving (FastAPI)"]
    CL["Client"] --> API["API layer<br/>auth, rate limit, validation"]
    API --> AB["Experiment assignment<br/>hash of user_id"]
    AB --> RC{"Redis cache"}
    RC -- hit --> OUT["Response"]
    RC -- miss --> ANN["ANN retrieval<br/>+ metadata filter"]
    ANN --> RR["Re-ranker<br/>popularity, recency, diversity"]
    RR --> OUT
    ANN -. index or DB down .-> FB["Fallback<br/>tier 3 (DB) then tier 4 (memory)"]
    FB --> OUT
  end

  IDXREG -- "active index_version" --> ANN
  API -.-> OBS["Prometheus, Grafana,<br/>OpenTelemetry, structlog"]
```

**Request flow**

1. Authenticate, rate-limit, and validate the request.
2. Assign the user to an experiment variant (deterministic hash of `user_id`) and write an exposure row (ADR-0017). A stopped or disabled experiment still produces an assignment with `log_exposure = false`, so the response carries the variant the user would have seen without touching the database.
3. Look up Redis. The cache key includes the active `index_version` and the endpoint, so index swaps never serve stale candidates (ADR-0015). The cache stores retrieval *candidates*, not the re-ranked list, so a hot-reload of the rerank config does not invalidate entries. If Redis is down, the cache is bypassed (a shared circuit breaker prevents repeated timeouts).
4. On a miss, run the filtered ANN query on the active index, then re-rank. On a hit, skip retrieve and run only the re-ranker.
5. If the ANN path fails (timeout, no active index, encoder unavailable, database unreachable), the fallback chain tries the popular-items snapshot (tier 3) and then the in-memory popularity cache (tier 4). A client error — empty seeds or all seeds missing — returns 400/404 and never enters the chain.
6. Return the result with `meta.source` (`cache`, `ann`, `fallback_ann`, `fallback_cached`, or `none`), and emit metrics, traces, logs, and the experiment exposure event. `model_version` and `index_version` are `null` in a fallback response: no model or index was used.

## Tech stack

| Layer | Choice |
|---|---|
| API | Python 3.11+ (CI tests 3.11 and 3.12), FastAPI, Pydantic v2 |
| Embeddings | sentence-transformers, exported to ONNX Runtime for fast CPU inference |
| Vector search | pgvector (HNSW) by default; FAISS as a benchmark option |
| Data and cache | PostgreSQL 16 + pgvector; Redis (cache and lightweight feature store) |
| Pipelines and versioning | Plain Python CLI (orchestration deferred to M6); versioned artifact runs; DVC or MLflow for the registry |
| Observability | Prometheus, Grafana, OpenTelemetry, structlog |
| Quality | pytest, Ruff, mypy, pre-commit, Locust or k6 |
| Infrastructure | Docker (multi-stage), docker-compose (dev), GitHub Actions, Terraform or Helm (optional) |
| Security | API key / JWT, rate limiting, Dependabot, Trivy |

## Project structure

```text
embedding-recommender-api/
├── src/recsys/
│   ├── api/                # routers, schemas, middleware, deps, readiness
│   │   ├── auth/           # API key + JWT (ADR-0013)
│   │   ├── middleware/     # request context, access log, rate limit
│   │   ├── routers/        # recommend, events, churn, health, metrics
│   │   ├── schemas/        # Pydantic v2 request/response models
│   │   ├── deps.py         # dependency providers (settings, cache, pool, principal)
│   │   ├── pipeline.py     # sync_retrieve / sync_rerank / sync_pipeline
│   │   └── readiness.py    # check_db / check_index / check_redis + aggregate
│   ├── embeddings/         # preprocess, encoder, ONNX export, pipeline, artifacts
│   ├── evaluation/         # metrics, golden_set, baselines, thresholds, runner
│   ├── retrieval/          # base, registry, filters, identity, pgvector,
│   │                       # numpy_backend, build, promote, rerank, providers,
│   │                       # active_index
│   ├── fallback/           # tier-3 / tier-4 chain (ADR-0020)
│   ├── experiments/        # assignment, loader, exposure, runtime (ADR-0017)
│   ├── events/             # batch event ingestion (ADR-0018)
│   ├── popularity/         # snapshot reader, in-memory cache, refresh
│   ├── cache/              # cache-aside store, key builder, TTL jitter (ADR-0015)
│   ├── resilience/         # circuit breaker (ADR-0015)
│   ├── rate_limit/         # token bucket in Redis via Lua (ADR-0014)
│   ├── churn/              # features, model, scoring (M7)
│   ├── monitoring/         # structlog, Prometheus registry, drift (M4)
│   └── config/             # Pydantic Settings, hot config store, enum validation
├── pipelines/              # orchestration wrappers (M6)
├── evaluation/
│   ├── golden_set/v1.jsonl # 20 queries, one per topic (ADR-0009)
│   ├── thresholds.yaml     # gate thresholds (ADR-0010)
│   ├── thresholds_history.yaml
│   └── report.json         # generated by `make eval`; not committed
├── migrations/             # Alembic env + versions (0001–0004)
├── data/
│   └── sample/             # sample catalog; real data via DVC
├── scripts/                # generate_sample_catalog, export_onnx, embed,
│                           # build_index, promote_index, rollback_index,
│                           # refresh_popularity, eval, bench_faiss,
│                           # render_benchmark, check_markers
├── tests/
│   ├── unit/               # fast, no external services
│   ├── integration/        # real PostgreSQL (pgvector) and Redis; encoder parity;
│   │                       # index lifecycle; backend agreement; determinism tiers;
│   │                       # auth wiring; end-to-end M3
│   └── load/               # Locust / k6 scenarios (M4)
├── deploy/
│   ├── docker/
│   ├── k8s/                # (M6)
│   └── terraform/          # (M6)
├── dashboards/             # Grafana JSON (M4)
├── docs/
│   ├── decisions.md        # pre-flight design decisions (locked)
│   ├── contracts.md        # data, API, config, telemetry contracts
│   ├── production-api.md   # M3 design doc
│   ├── embedding-pipeline.md       # M1 design
│   ├── retrieval-and-evaluation.md # M2 design
│   ├── runbook.md          # incident response procedures
│   ├── ops.md              # backup, restore, index lifecycle, disk planning
│   ├── faiss-benchmark.json        # raw measurement (ADR-0011)
│   ├── faiss-benchmark.md          # generated from the JSON
│   ├── api.md              # (M3)
│   └── adr/                # ADR-0001 through ADR-0025
├── .github/
│   ├── workflows/          # ci.yml (cd.yml, security.yml in M6/M8)
│   └── dependabot.yml      # (M8)
├── .env.example
├── .pre-commit-config.yaml
├── .dockerignore
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── alembic.ini
├── Makefile
├── CHANGELOG.md
└── LICENSE
```

## Quickstart

Prerequisites: Python 3.11+, GNU Make, PostgreSQL 16 with pgvector, and Redis (for the served path). Docker is optional and only needed for the container build.

```bash
git clone https://github.com/WoodinGlass/embedding-recommender-api.git
cd embedding-recommender-api

cp .env.example .env       # set API keys and database credentials

# One-time: install the extras the pipeline needs and produce the ONNX artifact.
pip install -e ".[export,pipeline,inference]"
python scripts/export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2

# Embed the sample catalog; writes a versioned run and updates `current`.
python -m recsys.embeddings.pipeline --mode=batch \
  --catalog data/sample/catalog.jsonl \
  --out artifacts/embeddings

# Apply migrations, build an index from the active run, and promote it.
export DATABASE_URL='postgresql://recsys:recsys@localhost:5432/recsys'
make migrate
make index-build
make index-promote VERSION="$(python -c "
import json, subprocess, sys
out = subprocess.run(['python', 'scripts/build_index.py'], capture_output=True, text=True)
print(json.loads(out.stdout.strip())['index_version'])
")"

# Run the offline evaluation and the threshold gate.
make eval
```

Request recommendations (use a key from `API_KEYS` in your `.env`):

```bash
curl -X POST http://localhost:8000/v1/recommend \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_123", "seed_item_ids": ["i_456"], "k": 10, "filters": {"category": "books"}}'
```

Interactive OpenAPI docs are served at `http://localhost:8000/docs`. Prometheus and Grafana land in M4.

## Configuration

Settings are read from environment variables (see `.env.example`). Never commit real secrets; `.env` is git-ignored. The full config contract — enum values, allowed ranges, and the prod-mode startup guards — lives in [`docs/contracts.md`](docs/contracts.md) § 3.

| Variable | Description | Example |
|---|---|---|
| `APP_ENV` | Selects the settings profile in `src/recsys/config/` | `dev` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://recsys:recsys@postgres:5432/recsys` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `API_KEYS` | Semicolon-separated API key hashes (Argon2id) or plaintext in dev | `dev-key-1;dev-key-2` |
| `API_KEYS_ADMIN` | Keys with the admin scope | — |
| `JWT_SECRET` | Signing secret when JWT auth is enabled | — |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` |
| `JWT_MAX_AGE_SECONDS` | `iat` freshness bound | `86400` |
| `EMBEDDING_MODEL` | sentence-transformers model exported to ONNX | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_ONNX_PATH` | Directory holding `model.onnx` and its sidecars | `artifacts/onnx/sentence-transformers__all-MiniLM-L6-v2` |
| `INDEX_BACKEND` | `pgvector` (default) or `faiss` (benchmark) | `pgvector` |
| `HNSW_EF_SEARCH` | Query-time recall/latency knob | `100` |
| `USER_ID_HASH_SALT` | HMAC salt for user id hashing in events and exposures | — |
| `USER_ID_HASH_SALT_VERSION` | Salt version (bumped on rotation) | `1` |
| `EXPERIMENT_DISABLED` | Kill switch: treat every experiment as stopped | `0` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector endpoint | `http://otel-collector:4317` |

Rerank weights, MMR flags, rate limits, and cache TTLs come from `config/hot.yaml` (ADR-0022), not from environment variables. The file is polled; a change takes effect without a restart.

## API

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/recommend` | API key / JWT | Top-k recommendations with metadata filters |
| `GET` | `/v1/items/{item_id}/similar` | API key / JWT | Item-to-item similarity |
| `POST` | `/v1/events` | API key / JWT | Log `impression`, `click`, and `conversion` events for experiments |
| `POST` | `/v1/churn/score` | API key / JWT | Churn-risk score for a user (extension, M7) |
| `GET` | `/livez` | None | Liveness |
| `GET` | `/healthz` | None | Liveness (alias of `/livez`) |
| `GET` | `/readyz` | None | Readiness: `db` (required), `index` (required), `encoder` (prod-required), `redis` (optional); three-state aggregate |
| `GET` | `/metrics` | Internal | Prometheus metrics |

Example response from `POST /v1/recommend`:

```json
{
  "request_id": "7c1f9e2a",
  "items": [
    { "item_id": "i_789", "score": 0.87, "rank": 1 },
    { "item_id": "i_321", "score": 0.84, "rank": 2 }
  ],
  "meta": {
    "source": "ann",
    "model_version": "minilm-onnx-v1+a3f9e021",
    "index_version": "idx-a3f9e021",
    "experiment": { "name": "rerank_mmr", "variant": "treatment" }
  }
}
```

In a fallback response (`meta.source` is `fallback_ann` or `fallback_cached`), `model_version` and `index_version` are `null`. The field is nullable rather than a magic string (`"n/a"`, `""`, `"fallback"`) so a metric can filter `WHERE model_version IS NULL` and a client can branch on it without parsing.

Errors: `401`/`403` for auth, `404` for an unknown seed item, `422` for validation, `429` with `Retry-After` when rate limited, and `503` only when every tier failed (`meta.source: "none"`). Interactive OpenAPI docs are served at `/docs`; a summary lives in `docs/api.md`.

## Embedding pipeline and index lifecycle

- **Batch and incremental runs.** Every item carries a content hash. Incremental runs embed only new or changed items; batch runs re-embed the whole catalog, for example after a model upgrade. A run with no changes exits 0 with `status="no_changes"` and leaves the active run untouched.
- **Versioned runs.** Each pipeline invocation writes its artifacts into `artifacts/embeddings/runs/<run_id>/` and updates the `current` pointer file **last**. `<run_id>` is `<ISO8601-Z>__<model_version>`. Rollback is a pointer swap; no file moves.
- **Atomic commit.** The commit order is fixed and every step is an atomic `os.replace`: Parquet, then `manifest.json`, then `state.json`, then `current`. A crash anywhere earlier leaves the previous run in place.
- **Locking.** Every run takes an exclusive `fcntl.flock` on `artifacts/embeddings/.lock`. `flock` is auto-released on process death. NFS is not supported (documented in `docs/adr/0004-versioned-runs-with-current-pointer.md`).
- **Streaming reuse.** Reuse does not load the previous Parquet into memory: the pipeline walks it with `ParquetFile.iter_batches` and writes the output through a `ParquetBatchWriter`, one row group per batch. Peak memory is O(batch_size × dim).
- **Pre-commit validation.** Every batch is checked for dtype, rank, row count, dimension, finiteness, unique ids, and L2 normalization before anything is written.
- **Deterministic by design.** Pinned model version, fixed preprocessing (`PREPROCESSING_VERSION`), stable item ordering, and pinned thread settings. The same catalog snapshot and model version produce the same embeddings. See [`docs/embedding-pipeline.md`](docs/embedding-pipeline.md) for the full three-tier contract.
- **Blue/green index swap.** A new index version is built next to the live one, evaluated against the golden set, and promoted by switching the active-version pointer. The previous version is kept for instant rollback. Index identity is content-addressed (`idx-<sha8>` over seven inputs; ADR-0007) so a rebuild with the same inputs is a no-op.
- **Active index caching.** The active index version and the pgvector extension version are read once at startup and refreshed on a 30 s timer (ADR-0015 amendment). The request path never queries `index_registry`; the versions travel into the pipeline as arguments.

```bash
# Produce or refresh the ONNX artifact (skips if the SHA already matches).
python scripts/export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2

# Batch embed the sample catalog; writes a new run and updates `current`.
python -m recsys.embeddings.pipeline --mode=batch \
  --catalog data/sample/catalog.jsonl \
  --out artifacts/embeddings

# Incremental: encodes only items whose content hash changed.
python -m recsys.embeddings.pipeline --mode=incremental \
  --catalog data/sample/catalog.jsonl \
  --out artifacts/embeddings
```

Stdout is a single JSON line (`pipeline.done` summary); per-batch progress
goes to stderr as JSON lines and can be silenced with `--quiet`. Exit codes:
`0` OK (including `no_changes`), `2` input error, `3` encoder error,
`4` output error, `5` lock timeout.

## Retrieval, re-ranking, and fallback

- **Retrieval.** pgvector HNSW with cosine distance on normalized embeddings. Filtered queries use `hnsw.iterative_scan = 'strict_order'` with a bounded `hnsw.max_scan_tuples` on pgvector 0.8+; older versions over-fetch by 4× and log a per-invocation warning. Filters go through a `FILTER_FIELDS` allowlist (`category`, `brand`, `language`) and parameterized queries, never string-built SQL. Backend identity and the filter strategy are in ADR-0007 and ADR-0008.
- **Re-ranking.** `score = w_sim * similarity + w_pop * popularity + w_rec * recency_decay`, followed by MMR for diversity. Weights live in `config/hot.yaml`, and each configuration is an experiment arm. The production default is the blend without MMR until MMR is proven (ADR-0016).
- **Cache-aside.** The cache stores retrieval *candidates* (item id, similarity, popularity, age_days) keyed on the active `index_version`, the endpoint, the filters, and the deduplicated seed set, at `cache_k_max` candidates per entry (default 100) sliced to the request's `k`. A hot-reload of rerank weights does not invalidate entries; an index swap does. MMR-enabled requests skip the cache in M3.6 (their candidates carry a 384-float vector per row) with a documented target design (ADR-0015 amendment).
- **Fallback chain (ADR-0020).** Four tiers plus 503:

| Tier | `meta.source` | Served when |
|---|---|---|
| 1 | `cache` | A cache hit for the exact request. The reranker still runs, so a config change takes effect on the next request. |
| 2 | `ann` | The filtered ANN query on the active index returns within the timeout and the re-ranker produces a list. |
| 3 | `fallback_ann` | The ANN path failed or timed out, but the database is reachable. A pre-computed popular list is read from `popularity_snapshot`, filtered, and returned. |
| 4 | `fallback_cached` | The database is unreachable. The same list, held in process memory and loaded from disk at startup, is filtered in memory and returned. |
| 5 | `none` (503) | Every tier produced an empty list or raised. |

Client errors — `empty_seeds` and `all_seeds_missing` — never enter the chain: a 400/404 must not be turned into a popular list.

## Offline evaluation

- **Golden set.** Query-to-relevant-items pairs in `evaluation/golden_set/v1.jsonl`, committed. The version is the filename (ADR-0009); the file is JSONL, one query per line. The committed set has 20 queries, one per topic, each with three seed items and seven relevant items drawn from the same topic cluster. Larger sets are versioned with DVC.
- **Retrieval quality.** Recall@k, NDCG@k, and MRR against the golden labels.
- **ANN fidelity.** Overlap with exact (brute-force) kNN, so index tuning is not mistaken for a relevance change.
- **Baselines.** Random (deterministic per query id), synthetic popularity (a placeholder until M5's event-based provider; ADR-0009 § 4), and exact kNN.
- **CI gate.** `make eval` writes a JSON report and fails if any metric drops below `evaluation/thresholds.yaml`. CI runs it on the committed golden set; the gate also checks report freshness, commit match, and that every expected system is present.

Results (measured on the sample catalog, 200 items, k=10; the `evaluation gate` CI job reproduces them via `make eval`):

| System | Recall@10 | NDCG@10 | MRR | ANN recall vs exact |
|---|---|---|---|---|
| Random baseline | 0.107 | 0.090 | 0.178 | n/a |
| Popularity (synthetic) | 0.057 | 0.044 | 0.089 | n/a |
| Exact kNN | 0.886 | 0.893 | 0.967 | 1.000 |
| pgvector HNSW | 0.886 | 0.893 | 0.967 | 1.000 |
| FAISS HNSW (benchmark) | 0.886 | 0.893 | 0.967 | 1.000 |
| pgvector HNSW + re-ranker | TBD (M4) | TBD (M4) | TBD (M4) | — |

Reading the table:

- **pgvector HNSW matches exact kNN exactly** on this catalog: same top-10 for every query, so ANN fidelity is 1.000. This is the backend-agreement property that ADR-0012 requires, confirmed on the real index.
- **The embedding model is roughly eight times better than random** on Recall@10 and much further ahead on MRR. The sample catalog's topic clusters are what the model is expected to find; the numbers say it does.
- **Synthetic popularity is worse than random here** — a property of the placeholder, not of popularity as a signal. The `PopularityProvider` interface (ADR-0009 § 4) is the seam M5 will swap for an event-based provider whose distribution actually resembles popularity. The M3.4 arms in the CI report sit in the `informational` list for that reason; only `provider_missing_rate_*` gates them.
- **Seeds are excluded from retrieved results** before metrics are computed. Without this, the seeds occupy ranks 1..N of every query (they are the nearest neighbours of their own mean) and MRR collapses to `1/(n_seeds + 1)` regardless of model quality.
- **FAISS is a benchmark, not a serving path.** Its row matches exact kNN on this catalog because HNSW is fully connected at every `ef_search` in the benchmark's grid; the recall/latency trade-off shows up on larger catalogs.

Thresholds are in `evaluation/thresholds.yaml`; every value above is at or above its threshold and its absolute floor (ADR-0010).

## A/B testing

- **Assignment.** Stateless and deterministic: `bucket = sha256(f"{effective_salt}:{user_id}") % 10_000`, mapped to variants by traffic allocation, with `effective_salt = f"{APP_ENV}:{experiment.salt}"`. Python's built-in `hash()` is deliberately not used because it is randomized per process. A per-experiment salt keeps assignments independent across experiments.
- **Exposure logging.** An exposure row is written for every served variant whose assignment says `log_exposure = true`. The write is idempotent on `sha256(f"exposure:{experiment}:{request_id}")`. Stopped and kill-switched experiments produce an assignment but do not write.
- **Planning.** A sample-size calculator takes the baseline rate, minimum detectable effect, significance level, and power.
- **Validity checks.** A sample ratio mismatch (SRM) check using a chi-square test (alert at p < 0.001) runs before any result is read.
- **Analysis.** Two-proportion z-test for rates, Welch's t-test for continuous metrics, confidence intervals, and Holm correction across multiple metrics. Analysis is fixed-horizon: results are read only after the planned sample size is reached.
- **Guardrails.** p95 latency, error rate, and fallback rate are compared per variant; a variant that breaches a guardrail is flagged regardless of primary-metric lift.
- **Proof it works.** `make ab-simulate` replays synthetic traffic with known effects. A/A runs should produce false positives at about the chosen significance level, and A/B runs with an injected lift should be detected at the planned power. *(M5)*

```bash
python -m recsys.experiments.analyze --experiment rerank_mmr
```

## Observability

- **Metrics** (Prometheus, `/metrics`): `recsys_request_duration_seconds` (histogram by route, source, and status), `recsys_cache_requests_total{result}`, `recsys_cache_skip_total{reason}`, `recsys_fallback_total{reason}`, `recsys_errors_total{type}`, `recsys_circuit_breaker_state{name}`, `recsys_active_index_info{index_version,model_version}`, `recsys_active_index_staleness_seconds`, `recsys_active_index_refresh_failures_total`, and `recsys_embedding_drift_score`. Histogram buckets include 0.2 s so the latency target is directly measurable.
- **p95 query:** `histogram_quantile(0.95, sum by (le) (rate(recsys_request_duration_seconds_bucket{route="/v1/recommend"}[5m])))`
- **Tracing and logs.** OpenTelemetry spans around cache, ANN query, re-rank, and fallback. structlog JSON logs carry `request_id` and `trace_id`, with no raw PII. All logs go to **stderr**; stdout is reserved for program output (the pipeline's JSON summary, the CLI's single-line results).
- **Drift.** Recent query and item embeddings are compared with a reference window using centroid cosine shift and PSI over the top PCA components. *(M4)*
- **Alerts.** p95 > 200 ms for 10 minutes; 5xx rate > 1% for 5 minutes; sustained fallback-rate spike; sharp drop in cache hit rate; drift score above threshold; SRM detected in a running experiment; active-index staleness above twice the refresh interval. *(M4)*
- **Dashboards.** Grafana JSON in `dashboards/` (service health, cache and fallback, drift, experiments). *(M4)*

The full metric and log-field contract — including cardinality guardrails and forbidden log fields — lives in [`docs/contracts.md`](docs/contracts.md) § 4.

## Performance targets and results

**Target:** p95 < 200 ms on 100k+ items at the target RPS (fixed in M4).

**Method for the served path (M4):** Locust scenarios in `tests/load/`. Latency comes from the server-side Prometheus histogram and is cross-checked against client-side percentiles. Warm-cache and cold-cache runs are reported separately, and every result records hardware, dataset size, and commit SHA.

```bash
make seed-synthetic N=100000   # (M4)
python -m recsys.embeddings.pipeline --mode=batch
make load-test                 # (M4)
```

**In-process library comparison (M2, ADR-0011).** The table below is the benchmark in `scripts/bench_faiss.py`, on the sample catalog (200 items × 384 dims) with `hnsw_m=16`, `hnsw_ef_construction=64`, single-threaded FAISS, and a 50-query warmup over 500 measured queries. It answers "which ANN library is faster on the same vectors", not "what is the served latency". The served latency is measured in M4 against a running service.

| Configuration | ef_search | p50 (ms) | p95 (ms) | p99 (ms) | QPS | ANN recall@10 vs exact |
|---|---|---|---|---|---|---|
| Exact kNN (numpy) | — | 0.0387 | 0.0584 | 2.0731 | 11 663 | 1.0000 |
| FAISS HNSW | 40 | 0.0547 | 0.1483 | 0.1964 | 14 819 | 1.0000 |
| FAISS HNSW | 80 | 0.0709 | 0.0871 | 0.1019 | 13 500 | 1.0000 |
| FAISS HNSW | 160 | 0.1015 | 1.3719 | 2.1651 | 4 670 | 1.0000 |
| FAISS HNSW | 320 | 0.1181 | 2.1470 | 2.1840 | 3 947 | 1.0000 |

Environment: Intel Xeon @ 2.20 GHz, 2 logical cores, 12.7 GiB RAM, Linux 6.6, Python 3.13.16, NumPy 2.1.3, FAISS 1.15.1. Full report in [`docs/faiss-benchmark.md`](docs/faiss-benchmark.md); raw measurement in [`docs/faiss-benchmark.json`](docs/faiss-benchmark.json).

On a catalog of 200 items, HNSW is fully connected at every `ef_search` in the grid, so the ANN-fidelity column is 1.0000 across the board: HNSW matches exact search. On a larger catalog the column would show a recall/latency trade-off. The latency jump at `ef_search >= 160` reflects that the value exceeds the catalog size (200), so the graph walk does more work than the data justifies; `ef_search <= catalog_size` is the range that matters in practice.

**pgvector's served latency is not in this table.** ADR-0011 explains why: pgvector's latency is measured server-side through the connection the API uses, not in-process. It is the measurement M4 publishes against the p95 < 200 ms target. The FAISS table above is the library-to-library comparison that motivates the ADR-0001 decision to use pgvector by default.

## Churn-risk extension

A separate router (`/v1/churn/*`) and package (`recsys.churn`) built on the same infrastructure: event log, Redis feature store, registry, and monitoring.

- **Features.** Recency, frequency, and engagement-trend features computed from the event log.
- **Model.** Baseline logistic regression, then gradient-boosted trees (scikit-learn). The better model is registered and served.
- **Labels and splits.** Churn means no activity within a configurable window. Train, validation, and test splits are time-based to prevent leakage.
- **Evaluation.** ROC-AUC and calibration (reliability curve, Brier score), reported in `docs/`.
- **Monitoring.** Feature and score drift appear on the same dashboards as embedding drift. *(M7)*

## Deployment and CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Pull request, push to `main` | Ruff, mypy, test-marker discipline; unit tests on a matrix of 3.11 and 3.12; integration tests against PostgreSQL (pgvector) and Redis; a dedicated encoder job that runs ONNX parity and the full three-tier determinism contract; a dedicated evaluation job that builds an index and runs `make eval`, uploading the report as a build artifact; Docker build and smoke test of `/healthz`, `/readyz`, `/metrics` |
| `cd.yml` | Push to `main` | Build and push the image (tagged with the git SHA), deploy to staging, run smoke tests and the eval gate, promote to production after approval, auto-rollback if readiness or SLO checks fail *(M6)* |
| `security.yml` | Pull request, nightly | Trivy scans (filesystem and image), dependency review *(M8)* |

- **Docker Hub authentication.** The three Docker-dependent jobs (`test-integration`, `evaluation`, `docker-build`) authenticate to Docker Hub before pulling. An anonymous pull rate limit (100 / 6 h / IP) is not enough for a busy CI queue; an authenticated account raises it to 200 / 6 h. Service containers cannot use `docker/login-action` (they pull before any step runs), so the integration and evaluation jobs start PostgreSQL and Redis with explicit `docker run` after the login step.
- **Image.** Multi-stage Dockerfile; builder installs into a virtualenv, runtime copies only the virtualenv and runs as uid 1000 `recsys`.
- **Orchestration.** Rolling updates with readiness probes (`deploy/k8s`). Terraform (`deploy/terraform`) is optional.
- **Rollback.** Code rollback redeploys the previous image tag. Embedding-run rollback is a pointer swap in `artifacts/embeddings/current`. Index rollback is independent (`make index-rollback`).
- **Dependencies.** Dependabot is configured in `.github/dependabot.yml` *(M8)*.

## Security

- Authentication via API key (`X-API-Key`, Argon2id-hashed in prod) or JWT; privileged operations require an admin scope (ADR-0013).
- Redis-backed rate limiting per key, degrading to a per-instance limiter if Redis is down (ADR-0014).
- Strict Pydantic validation: bounded `k`, bounded list sizes, and an allowlist of filter fields.
- Filter values always reach SQL as parameters; the field set is fixed by `FILTER_FIELDS` (ADR-0008).
- User ids in events and exposures are stored as HMAC-SHA256 under a versioned salt; the raw id never reaches storage, logs, or metrics (ADR-0018).
- Secrets only through environment variables; least-privilege database user.
- Trivy scans in CI and automated dependency updates through Dependabot.
- Logs contain no raw PII.

## Development and testing

Requires Python 3.11+ and GNU Make. PostgreSQL 16 with pgvector and Redis are needed for the served path and the integration tests.

```bash
make install-dev          # pip install -e ".[dev]" — the full local environment
make install-hooks        # pre-commit install --install-hooks
make check                # lint + typecheck + check-markers + unit tests
```

**Pre-commit hooks.** `make install-hooks` writes a `pre-commit` hook to
`.git/hooks/` and downloads the hook environments. On every `git commit`,
the hooks run ruff (lint + format), mypy, and a small set of
housekeeping checks (trailing whitespace, YAML/TOML syntax, large
files). A fixable failure (ruff) is corrected in place and the commit is
aborted with the fix staged; a non-fixable failure (mypy) aborts and
prints the reason. `git commit --no-verify` skips the hook for one
commit; that is the escape hatch, not the workflow.

The ruff version is pinned in three places that must stay equal:
`pyproject.toml` (what `make lint` uses), `.pre-commit-config.yaml`
(what the hook uses), and the version CI installs. A commit that passes
with one ruff and fails with another is a class of failure the project
already spent time on; the pin removes it. The pin is checked
mechanically by the lint job.

**In Colab.** Hook environments live in `~/.cache/pre-commit/`, which
is per-session; the hook itself (`.git/hooks/pre-commit`) persists in
Drive. The first commit of a new session downloads the environments
(~30 s); later commits are fast. If you do not plan to commit in this
session, `make install-hooks` is not required.

Extras are provided so CI and local development do not pay for what they do not use:

| Extra | Contents | When to use |
|---|---|---|
| `[dev-lite]` | API + observability + test/quality tooling. **No torch.** | CI lint and unit jobs; fast local iteration. |
| `[inference]` | numpy + onnxruntime + tokenizers. **No torch.** | Serving path, and anything that loads a `.onnx` artifact. |
| `[export]` | sentence-transformers + onnx. Pulls torch. | Producing a new ONNX artifact; the encoder parity test. |
| `[pipeline]` | `[inference]` + pyarrow. | The batch/incremental embedding pipeline and artifact I/O. |
| `[db]` | SQLAlchemy + psycopg + Alembic + pgvector. | Migrations, the pgvector backend, the build/promote CLIs. |
| `[cache]` | redis-py (async). | The cache store, the rate limiter, and the readiness check. |
| `[auth]` | passlib[argon2] + PyJWT. | API key and JWT validation. |
| `[config]` | pyyaml. | Loading `config/hot.yaml` and `experiments.yaml`. |
| `[bench]` | faiss-cpu. | `scripts/bench_faiss.py` only. Never a production dependency. |
| `[dev]` | Superset of `[dev-lite]` plus embeddings, experiments, churn, bench, and load. | Full local development. |

| Command | What it does |
|---|---|
| `make fmt` | Ruff auto-fix and format |
| `make lint` | Ruff check and format-check, no modifications |
| `make typecheck` | mypy in strict mode |
| `make check-markers` | Enforce test-tier discipline (unit tests must not require external services) |
| `make test` | Unit + integration tests (integration skips without `RECSYS_TEST_*` env) |
| `make test-unit` | Unit tests only |
| `make test-integration` | Integration tests, light tier (PostgreSQL / Redis); skips encoder parity |
| `make test-encoder` | Encoder parity and the three-tier determinism contract. Requires `[inference,export,pipeline]` |
| `make coverage` | Unit tests with coverage report |
| `make eval` | Offline evaluation + threshold gate (ADR-0010) |
| `make migrate` | Apply Alembic migrations (reads `DATABASE_URL` from the environment) |
| `make index-build` | Build an index from the active embedding run |
| `make index-promote VERSION=<v>` | Promote an index; `make index-rollback` reverts to the most recent retired one |
| `make popularity-refresh` | Refresh the popularity snapshot and rewrite the tier-4 cache file |
| `make bench-faiss` | Run the FAISS benchmark and render the Markdown companion (ADR-0011) |
| `make seed` | Load the sample catalog; `make seed-synthetic N=100000` generates one *(M4)* |
| `make load-test` | Locust load test against the local stack *(M4)* |
| `make ab-simulate` | Simulated A/A and A/B experiments with known effects *(M5)* |
| `make clean` | Remove caches and build artifacts |

Test layers:

- `tests/unit` covers pure logic: preprocessing rules, encoder protocol conformance, artifact primitives (locking, atomic writes, config hashing, run-id allocation, state validation, Parquet I/O), catalog loading, mode planning, retrieval metrics, golden set loading, evaluation thresholds, evaluation runner, the retrieval backends with fake connections, the auth dependency against a minimal FastAPI app, the readiness aggregate, the fallback chain with a monkeypatched reader, the experiment runtime, and the recommend handler's mapping from a pipeline outcome to an HTTP response.
- `tests/integration` runs against real services: PostgreSQL (pgvector) and Redis; ONNX-vs-reference encoder parity; the three-tier determinism contract; the index lifecycle (build, promote, rollback, incomplete-build refusal, at-most-one-active invariant); the backend-agreement regression test from ADR-0012; auth wiring across every protected endpoint; and an end-to-end M3 test that boots the app factory, middleware stack, auth, connection pool, readiness checks, and the recommend / events / similar handlers against a database with migrations applied and no active index. These skip cleanly unless the relevant env vars or extras are present.
- `tests/load` holds the Locust (or k6) scenarios for the served-path latency target *(M4)*.

Pull requests must pass CI. Architectural changes need an ADR in `docs/adr/` — see the [ADR index](docs/adr/README.md) for the convention and the **twenty-five accepted records** covering M0 through M3 (pgvector as default, ONNX Runtime for inference, plain Python CLI for the pipeline, versioned runs with an atomic current pointer, M2 scope, the pgvector schema, index identity, filter strategy, golden set and metrics, evaluation thresholds, FAISS benchmark methodology, backend abstraction, authentication, rate limiting, the cache and circuit breaker, the re-ranker composition, experiment assignment, event ingestion, the readiness contract, the five-tier fallback chain, the observability contract, config management, deployment, SLOs, and the development environment).

## Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| pgvector as the default vector store | One datastore for vectors and metadata, transactional filtering, simple operations | Lower throughput ceiling and fewer tuning options than dedicated engines; the FAISS benchmark quantifies the gap |
| HNSW over IVFFlat | Better recall/latency trade-off and no training step | Higher memory use and slower index builds |
| ONNX Runtime for encoding | Faster CPU inference and a smaller runtime footprint | Extra export step; parity with the original model is verified in tests |
| Plain Python CLI for the pipeline | Debuggable anywhere, stable integration surface, no scheduler to install | No retries or scheduling until M6, when an orchestrator wraps the CLI |
| Versioned runs with an atomic `current` pointer | A crash cannot leave a state file pointing at a missing artifact; rollback is a pointer swap | Old runs accumulate on disk until a prune target is added |
| Content-addressed `index_version` (`idx-<sha8>`) | A rebuild with the same inputs is a no-op; a change to any build input produces a different id | An operator changing a build parameter must publish the new id; a rebuild is required |
| Seeds excluded from retrieved results | A recommender does not recommend what the user already has; without this, MRR collapses to `1/(n_seeds+1)` | Over-fetch by `k + len(seeds)` and filter in the caller; the backend protocol stays small |
| Cache stores retrieval candidates, not responses | A hot-reload of rerank weights must not invalidate every entry; only index swaps should | Cache hits still pay the re-rank cost. A cache miss stores a `k_max`-sized candidate window (100), so a `k=5` request retrieves more than it strictly needs; `cache_k_max` is tunable |
| Redis cache keyed by index version and endpoint | Index swaps never serve stale neighbors; the two endpoints never share entries | Staleness within the TTL; one more moving part (bypassed on failure) |
| Fallback instead of failing | Availability over freshness; the process memory tier survives a database outage that also takes Redis down | Lower relevance while degraded, tracked through fallback rate and guardrails; a fallback response carries `null` versions rather than pretending to have a model |
| Hash-based experiment assignment | Stateless, reproducible, consistent across instances | No dynamic re-allocation without re-bucketing users |
| Blue/green indexes | Zero downtime and instant rollback | Roughly double the index storage during a swap |
| Offline evaluation as a CI gate | Catches regressions before deploy | Offline metrics do not guarantee online lift, which is why A/B testing exists |
| Async handler, one thread hop per request | The pipeline is synchronous by design (ADR-0012); wrapping each step separately pays four hops for no concurrency gain | A bounded thread pool (`CapacityLimiter`, default 20) caps throughput under load; the queue is observable, a saturated database is not |

For the full reasoning behind these choices — including the alternatives that
were considered and rejected — see the ADRs under [`docs/adr/`](docs/adr/).

## Milestones

| # | Milestone | Scope | Exit criteria | Status |
|---|---|---|---|---|
| M0 | Foundation | Repo, CI (lint, type check, test), Docker, pre-commit, first ADR | CI is green on the scaffold; the `docker-build` job builds the image and serves `/healthz`, `/readyz`, and `/metrics` in a container | Done |
| M1 | Embedding pipeline | Batch and incremental embedding, model/index versioning, golden set | Re-running the pipeline produces identical results | Done |
| M2 | Retrieval and offline evaluation | pgvector HNSW, benchmark vs FAISS, Recall@k / NDCG / MRR | Metrics are documented and enforced as a CI gate | Done |
| M3 | Production API | Auth, rate limiting, caching, fallback, health checks, re-ranker arm | Integration tests are green | Done |
| M4 | Observability and load test | Prometheus/Grafana, tracing, Locust | p95 < 200 ms at the target RPS, with evidence committed in `docs/` | Planned |
| M5 | A/B testing | Assignment, logging, statistical analysis, dashboard | Simulation reaches the correct conclusion on a known effect | Planned |
| M6 | Deployment | Automated CD, staging to prod, rollback, blue/green index | A merge to `main` reaches staging automatically; production promotion, rollback, and index swap are demonstrated with no downtime | Planned |
| M7 | Churn extension | Features, model, endpoint, drift monitoring | Endpoint serves scores; AUC and calibration documented; drift monitored | Planned |
| M8 | Hardening | Security scan, runbook, simulated postmortem, final README | No unaddressed high/critical findings; runbook and postmortem published; README documents architecture and trade-offs | Planned |

> **Note on the M0 exit criteria.** The primary development environment for
> this project is Google Colab, which has no Docker daemon. "Serves `/healthz`
> in a container" is therefore proven in the CI `docker-build` job's smoke test
> (which starts the freshly built image and curls `/healthz`, `/readyz`, and
> `/metrics`), not by a local `make up`.

> **Note on the M1 exit criteria.** "Identical results" is enforced at three
> tiers, defined in [`docs/embedding-pipeline.md`](docs/embedding-pipeline.md) § 5
> and tested in `tests/integration/test_determinism_tiers.py`: byte-identical
> Parquet in the pinned CI environment (strict), identical top-k neighbours
> everywhere (semantic), and per-row cosine similarity ≥ 0.9999 (tolerance).
> Only the strict tier is CI-only; the other two run on every `make test-encoder`.

> **Note on the M2 exit criteria.** "Metrics are documented and enforced as a
> CI gate" is proven in two places. Documentation: the evaluation table above,
> filled with measured values from the CI `evaluation gate` job; the definitions
> are in `docs/adr/0009-golden-set-and-metrics.md` and the rationale for the
> thresholds in `docs/adr/0010-evaluation-thresholds.md`. Enforcement: the
> `evaluation` job fails the build when any metric is below its threshold or
> its absolute floor, when the report's golden set version does not match the
> threshold file's, or when a required system is missing from the report.

> **Note on the M3 exit criteria.** "Integration tests are green" is proven
> by the CI `integration` job, which provisions real PostgreSQL (pgvector)
> and Redis and runs `tests/integration/`. The M3 additions to that suite
> are `test_auth_wiring.py` (15 tests: every protected endpoint rejects
> without a credential and accepts a valid one, every open endpoint stays
> open) and `test_m3_end_to_end.py` (real app factory, middleware stack,
> auth, connection pool, readiness checks, and the recommend / events /
> similar handlers against a database with migrations applied and no active
> index). The evaluation gate remains separate and continues to be the M2
> exit criterion.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).
