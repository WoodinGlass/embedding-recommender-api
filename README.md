# embedding-recommender-api

[![CI](https://github.com/WoodinGlass/embedding-recommender-api/actions/workflows/ci.yml/badge.svg)](https://github.com/WoodinGlass/embedding-recommender-api/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Embedding-based recommendation service with low-latency ANN retrieval (target p95 < 200 ms), re-ranking, statistically valid A/B testing, monitoring, and automated deployment (Docker + CI/CD). It ships with offline evaluation, model/index versioning, fallbacks, and a churn-risk extension.

> **Status:** in development. M0 (Foundation) and M1 (Embedding pipeline) are complete; see [Milestones](#milestones). Performance figures are targets until the M4 benchmark is published.

## What this is / is not

**This is**

- A production-style recommendation service: embed a catalog, retrieve neighbors with ANN search and metadata filters, re-rank, and serve the result through FastAPI.
- A reference for the full ML-serving lifecycle: versioned models and indexes, offline evaluation as a CI gate, A/B experimentation, observability, graceful degradation, and zero-downtime index swaps.
- A single-node system sized for catalogs of 100k+ items that runs locally with `make up`.
- A foundation reused for a second use case: a churn-risk scoring endpoint on the same infrastructure.

**This is not**

- A chatbot or RAG system. Embeddings are used for item recommendation, not text generation.
- A vector database. pgvector and FAISS sit behind a small retrieval interface; the project does not try to replace Milvus, Qdrant, or Pinecone.
- A model-training project. It uses pre-trained sentence-transformers models without fine-tuning. The churn model is the only model trained here.
- A web-scale system. Sharding, multi-region replication, and catalogs of 10M+ items are out of scope.
- A turnkey product. There is no admin UI, multi-tenancy, or billing.

## Features

1. **Embedding pipeline** — batch and incremental catalog embedding with versioned models and indexes, atomic commit, and a three-tier determinism contract.
2. **Similarity search with metadata filters** — target p95 < 200 ms on 100k+ items.
3. **Re-ranking** — popularity, recency, and diversity (MMR).
4. **Graceful degradation** — cold-start and fallback paths when the index or Redis is down.
5. **Offline evaluation in CI** — Recall@k, NDCG, and MRR with thresholds that fail the build.
6. **A/B testing** — deterministic assignment (hashed user ID), SRM check, sample-size calculation, significance tests, and guardrail metrics.
7. **API hardening** — authentication, rate limiting, input validation, and health/readiness endpoints.
8. **Observability** — latency, cache hit rate, error rate, embedding drift, and alerts.
9. **Churn-risk extension** — separate endpoint, evaluated with AUC and calibration.
10. **Zero-downtime index swap** — blue/green indexes with instant rollback.

## Architecture

```mermaid
flowchart TB
  subgraph OFFLINE["Offline / batch (plain Python CLI; orchestration lands in M6)"]
    CAT[("Catalog")] --> EMB["Embed job<br/>ONNX encoder"]
    EMB --> IDX["Build HNSW index<br/>(new version)"]
    IDX --> EVAL["Offline eval gate<br/>Recall@k, NDCG, MRR"]
    EVAL --> REG[("Model + index registry")]
  end

  subgraph ONLINE["Online serving (FastAPI)"]
    CL["Client"] --> API["API layer<br/>auth, rate limit, validation"]
    API --> AB["Experiment assignment<br/>hash of user_id"]
    AB --> RC{"Redis cache"}
    RC -- hit --> OUT["Response"]
    RC -- miss --> ANN["ANN retrieval<br/>+ metadata filter"]
    ANN --> RR["Re-ranker<br/>popularity, recency, diversity"]
    RR --> OUT
    ANN -. index or DB down .-> FB["Fallback<br/>popularity, cold-start"]
    FB --> OUT
  end

  REG -- "promote (blue/green)" --> ANN
  API -.-> OBS["Prometheus, Grafana,<br/>OpenTelemetry, structlog"]
```

**Request flow**

1. Authenticate, rate-limit, and validate the request.
2. Assign the user to an experiment variant (deterministic hash of `user_id`).
3. Look up Redis. The cache key includes `index_version` and the experiment variant, so index swaps and A/B arms never share entries. If Redis is down, the cache is bypassed instead of failing the request.
4. On a miss, run the filtered ANN query on the active index, then re-rank.
5. If the index or database is unavailable, or the ANN timeout is hit, serve the popularity/cold-start fallback.
6. Return the result with `meta.source` (`cache`, `ann`, or `fallback`), and emit metrics, traces, logs, and the experiment exposure event.

## Tech stack

| Layer | Choice |
|---|---|
| API | Python 3.11+ (CI tests 3.11 and 3.12), FastAPI, Pydantic v2 |
| Embeddings | sentence-transformers, exported to ONNX Runtime for fast CPU inference |
| Vector search | pgvector (HNSW) by default; FAISS as a benchmark option |
| Data and cache | PostgreSQL; Redis (cache and lightweight feature store) |
| Pipelines and versioning | Plain Python CLI (orchestration deferred to M6); versioned artifact runs; DVC or MLflow for the registry |
| Observability | Prometheus, Grafana, OpenTelemetry, structlog |
| Quality | pytest, Ruff, mypy, pre-commit, Locust or k6 |
| Infrastructure | Docker (multi-stage), docker-compose (dev), GitHub Actions, Terraform or Helm (optional) |
| Security | API key / JWT, rate limiting, Dependabot, Trivy |

## Project structure

```text
embedding-recommender-api/
├── src/recsys/
│   ├── api/                # routers, schemas, middleware (auth, rate limit)
│   ├── embeddings/         # preprocess, encoder, ONNX export, pipeline, artifacts
│   ├── retrieval/          # ANN index, filtering, re-ranker (interface only; M2)
│   ├── fallback/           # popularity and cold-start
│   ├── experiments/        # assignment, logging, stats analysis
│   ├── churn/              # features, model, scoring
│   ├── monitoring/         # metrics, drift detection
│   └── config/             # settings per environment
├── pipelines/              # orchestration wrappers (M6)
├── evaluation/             # offline eval (Recall@k, NDCG, MRR), golden set
├── migrations/             # Alembic (M2)
├── data/
│   └── sample/             # sample catalog; real data via DVC
├── scripts/                # generate_sample_catalog, export_onnx, embed, check_markers
├── tests/
│   ├── unit/               # fast, no external services
│   ├── integration/        # real PostgreSQL (pgvector) and Redis; encoder parity
│   └── load/               # Locust / k6 scenarios
├── deploy/
│   ├── docker/
│   ├── k8s/
│   └── terraform/
├── dashboards/             # Grafana JSON
├── docs/
│   ├── decisions.md        # pre-flight design decisions (locked)
│   ├── contracts.md        # data, API, config, and telemetry contracts
│   ├── embedding-pipeline.md  # M1 design: layout, atomicity, determinism tiers
│   ├── runbook.md          # incident response procedures
│   ├── api.md              # API summary (planned)
│   └── adr/                # architecture decision records
├── .github/
│   ├── workflows/          # ci.yml (cd.yml, security.yml planned)
│   └── dependabot.yml      # planned
├── .env.example
├── .pre-commit-config.yaml
├── .dockerignore
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── CHANGELOG.md
└── LICENSE
```

## Quickstart

Prerequisites: Docker with Compose v2, GNU Make, and Python 3.11+ (for local development).

```bash
git clone https://github.com/WoodinGlass/embedding-recommender-api.git
cd embedding-recommender-api

cp .env.example .env       # set API keys and database credentials

# One-time: install the export extra and produce the ONNX artifact.
pip install -e ".[export,pipeline,inference]"
python scripts/export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2

# Embed the sample catalog and activate the first run.
python -m recsys.embeddings.pipeline --mode=batch \
  --catalog data/sample/catalog.jsonl \
  --out artifacts/embeddings

# The full local stack (API, PostgreSQL with pgvector, Redis, Prometheus,
# Grafana) starts in M3 once the retrieval path lands.
make up                    # (planned until M3)
make migrate               # (planned until M2)
```

Request recommendations (use a key from `API_KEYS` in your `.env`):

```bash
curl -X POST http://localhost:8000/v1/recommend \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u_123", "seed_item_ids": ["i_456"], "k": 10, "filters": {"category": "books"}}'
```

Local endpoints: API docs at `http://localhost:8000/docs`, Prometheus on `:9090`, Grafana on `:3000`.

## Configuration

Settings are read from environment variables (see `.env.example`). Never commit real secrets; `.env` is git-ignored. The full config contract — enum values, allowed ranges, and the prod-mode startup guards — lives in [`docs/contracts.md`](docs/contracts.md) § 3.

| Variable | Description | Example |
|---|---|---|
| `APP_ENV` | Selects the settings profile in `src/recsys/config/` | `dev` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://recsys:recsys@postgres:5432/recsys` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `API_KEYS` | Comma-separated API keys (or use JWT instead) | `dev-key-1` |
| `JWT_SECRET` | Signing secret when JWT auth is enabled | — |
| `RATE_LIMIT_PER_MINUTE` | Per-key request limit | `600` |
| `EMBEDDING_MODEL` | sentence-transformers model exported to ONNX | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_ONNX_PATH` | Directory holding `model.onnx` and its sidecars | `artifacts/onnx/sentence-transformers__all-MiniLM-L6-v2` |
| `EMBEDDING_BATCH_SIZE` | Encode batch size for the pipeline | `64` |
| `INDEX_BACKEND` | `pgvector` (default) or `faiss` (benchmark) | `pgvector` |
| `HNSW_M`, `HNSW_EF_CONSTRUCTION` | Index build parameters | `16`, `64` |
| `HNSW_EF_SEARCH` | Query-time recall/latency knob | `100` |
| `ANN_TIMEOUT_MS` | Hard timeout before falling back | `120` |
| `CACHE_TTL_SECONDS` | Redis cache TTL | `300` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry collector endpoint | `http://otel-collector:4317` |

## API

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/v1/recommend` | API key / JWT | Top-k recommendations with metadata filters |
| `GET` | `/v1/items/{item_id}/similar` | API key / JWT | Item-to-item similarity |
| `POST` | `/v1/events` | API key / JWT | Log `impression`, `click`, and `conversion` events for experiments |
| `POST` | `/v1/churn/score` | API key / JWT | Churn-risk score for a user (extension) |
| `GET` | `/healthz` | None | Liveness |
| `GET` | `/readyz` | None | Readiness: database reachable and an active index loaded (Redis status is reported, not required) |
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
    "index_version": "idx-0003",
    "experiment": { "name": "rerank_mmr", "variant": "treatment" }
  }
}
```

Errors: `401`/`403` for auth, `422` for validation, `429` with `Retry-After` when rate limited, and `503` only when both ANN retrieval and the fallback fail. Interactive OpenAPI docs are served at `/docs`; a summary lives in `docs/api.md`.

The recommend/events/churn endpoints currently return `503` with a milestone
message. The schemas are final; implementations land in M3, M5, and M7.

## Embedding pipeline and index lifecycle

- **Batch and incremental runs.** Every item carries a content hash. Incremental runs embed only new or changed items; batch runs re-embed the whole catalog, for example after a model upgrade. A run with no changes exits 0 with `status="no_changes"` and leaves the active run untouched.
- **Versioned runs.** Each pipeline invocation writes its artifacts into `artifacts/embeddings/runs/<run_id>/` and updates the `current` pointer file **last**. `<run_id>` is `<ISO8601-Z>__<model_version>`. Rollback is a pointer swap; no file moves.
- **Atomic commit.** The commit order is fixed and every step is an atomic `os.replace`: Parquet, then `manifest.json`, then `state.json`, then `current`. A crash anywhere earlier leaves the previous run in place.
- **Locking.** Every run takes an exclusive `fcntl.flock` on `artifacts/embeddings/.lock`. `flock` is auto-released on process death. NFS is not supported (documented in `docs/adr/0004-versioned-runs-with-current-pointer.md`).
- **Streaming reuse.** Reuse does not load the previous Parquet into memory: the pipeline walks it with `ParquetFile.iter_batches` and writes the output through a `ParquetBatchWriter`, one row group per batch. Peak memory is O(batch_size × dim).
- **Pre-commit validation.** Every batch is checked for dtype, rank, row count, dimension, finiteness, unique ids, and L2 normalization before anything is written.
- **Deterministic by design.** Pinned model version, fixed preprocessing (`PREPROCESSING_VERSION`), stable item ordering, and pinned thread settings. The same catalog snapshot and model version produce the same embeddings. See [`docs/embedding-pipeline.md`](docs/embedding-pipeline.md) for the full three-tier contract.
- **Blue/green index swap.** A new index version is built next to the live one, evaluated against the golden set, warmed up, and then promoted by switching the active-version pointer. API instances pick up the new pointer without a restart, and the previous version is kept for instant rollback. *(M2)*

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

- **Retrieval.** pgvector HNSW with cosine distance on normalized embeddings. Build parameters (`m`, `ef_construction`) and the query-time `hnsw.ef_search` are tuned in M2, and the recall/latency trade-off is documented. For very selective filters, use iterative index scans (pgvector 0.8+), partial indexes, or partitioning so filtered queries still return `k` results. Filters go through an allowlist and parameterized queries, never string-built SQL.
- **Re-ranking.** `score = w_sim * similarity + w_pop * popularity + w_rec * recency_decay`, followed by MMR for diversity. Weights live in config, and each configuration can be an experiment arm.
- **Fallback chain.**

| Situation | Behavior | `meta.source` |
|---|---|---|
| Cache hit | Serve from Redis | `cache` |
| Redis unavailable | Bypass the cache; a circuit breaker prevents repeated timeouts | `ann` |
| ANN timeout, index or DB unavailable | Popularity-ranked items within the requested filters | `fallback` |
| Cold-start user (no seed items) or item without an embedding | Popularity and recency within the filters | `fallback` |

## Offline evaluation

- **Golden set.** Query-to-relevant-items pairs in `evaluation/golden_set/v1.jsonl`, committed. The version is the filename (ADR-0009); the file is JSONL, one query per line. Larger sets are versioned with DVC. The committed set has 20 queries, one per topic, each with three seed items and seven relevant items drawn from the same topic cluster.
- **Retrieval quality.** Recall@k, NDCG@k, and MRR against the golden labels.
- **ANN fidelity.** Overlap with exact (brute-force) kNN, so index tuning is not mistaken for a relevance change.
- **CI gate.** `make eval` writes a JSON report and fails if any metric drops below `evaluation/thresholds.yaml`. CI runs it on the committed golden set to stay fast; the full set runs before every index promotion. *(M2)*

Results (filled in during M2; only measured numbers belong here):

| System | Recall@10 | NDCG@10 | MRR | ANN recall vs exact |
|---|---|---|---|---|
| Popularity baseline | TBD | TBD | TBD | n/a |
| pgvector HNSW | TBD | TBD | TBD | TBD |
| pgvector HNSW + re-ranker | TBD | TBD | TBD | TBD |
| FAISS HNSW | TBD | TBD | TBD | TBD |

## A/B testing

- **Assignment.** Stateless and deterministic: `bucket = sha256(f"{experiment_salt}:{user_id}") % 10_000`, mapped to variants by traffic allocation. Python's built-in `hash()` is deliberately not used because it is randomized per process. A per-experiment salt keeps assignments independent across experiments.
- **Logging.** Exposure events (when a user is actually served a variant) and outcome events (`click`, `conversion`) are stored in PostgreSQL with `experiment`, `variant`, and `request_id`.
- **Planning.** A sample-size calculator takes the baseline rate, minimum detectable effect, significance level, and power.
- **Validity checks.** A sample ratio mismatch (SRM) check using a chi-square test (alert at p < 0.001) runs before any result is read.
- **Analysis.** Two-proportion z-test for rates, Welch's t-test for continuous metrics, confidence intervals, and Holm correction across multiple metrics. Analysis is fixed-horizon: results are read only after the planned sample size is reached.
- **Guardrails.** p95 latency, error rate, and fallback rate are compared per variant; a variant that breaches a guardrail is flagged regardless of primary-metric lift.
- **Proof it works.** `make ab-simulate` replays synthetic traffic with known effects. A/A runs should produce false positives at about the chosen significance level, and A/B runs with an injected lift should be detected at the planned power. *(M5)*

```bash
python -m recsys.experiments.analyze --experiment rerank_mmr
```

## Observability

- **Metrics** (Prometheus, `/metrics`): `recsys_request_duration_seconds` (histogram by route, source, and status), `recsys_cache_requests_total{result}`, `recsys_fallback_total{reason}`, `recsys_errors_total{type}`, `recsys_embedding_drift_score`, and `recsys_active_index_info{index_version,model_version}`. Histogram buckets include 0.2 s so the latency target is directly measurable.
- **p95 query:** `histogram_quantile(0.95, sum by (le) (rate(recsys_request_duration_seconds_bucket{route="/v1/recommend"}[5m])))`
- **Tracing and logs.** OpenTelemetry spans around cache, ANN query, re-rank, and fallback. structlog JSON logs carry `request_id` and `trace_id`, with no raw PII.
- **Drift.** Recent query and item embeddings are compared with a reference window using centroid cosine shift and PSI over the top PCA components. *(M4)*
- **Alerts.** p95 > 200 ms for 10 minutes; 5xx rate > 1% for 5 minutes; sustained fallback-rate spike; sharp drop in cache hit rate; drift score above threshold; SRM detected in a running experiment. *(M4)*
- **Dashboards.** Grafana JSON in `dashboards/` (service health, cache and fallback, drift, experiments). *(M4)*

The full metric and log-field contract — including cardinality guardrails and forbidden log fields — lives in [`docs/contracts.md`](docs/contracts.md) § 4.

## Performance targets and results

**Target:** p95 < 200 ms on 100k+ items at the target RPS (fixed in M4).

**Method:** Locust scenarios in `tests/load/`. Latency comes from the server-side Prometheus histogram and is cross-checked against client-side percentiles. Warm-cache and cold-cache runs are reported separately, and every result records hardware, dataset size, and commit SHA.

```bash
make seed-synthetic N=100000   # planned
python -m recsys.embeddings.pipeline --mode=batch
make load-test                 # planned
```

Results (filled in during M4):

| Configuration | Items | RPS | p50 | p95 | p99 | ANN recall@10 vs exact |
|---|---|---|---|---|---|---|
| pgvector HNSW | 100k | TBD | TBD | TBD | TBD | TBD |
| pgvector HNSW + Redis cache | 100k | TBD | TBD | TBD | TBD | TBD |
| FAISS HNSW (benchmark) | 100k | TBD | TBD | TBD | TBD | TBD |

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
| `ci.yml` | Pull request, push to `main` | Ruff, mypy, test-marker discipline; unit tests on a matrix of 3.11 and 3.12; integration tests against PostgreSQL (pgvector) and Redis service containers; a dedicated encoder job that runs ONNX parity and the full three-tier determinism contract; Docker build and smoke test of `/healthz`, `/readyz`, `/metrics` |
| `cd.yml` | Push to `main` | Build and push the image (tagged with the git SHA), deploy to staging, run smoke tests and the eval gate, promote to production after approval, auto-rollback if readiness or SLO checks fail *(planned)* |
| `security.yml` | Pull request, nightly | Trivy scans (filesystem and image), dependency review *(planned)* |

- **Image.** Multi-stage Dockerfile; builder installs into a virtualenv, runtime copies only the virtualenv and runs as uid 1000 `recsys`. `docker-compose.yml` is for local development only.
- **Orchestration.** Rolling updates with readiness probes (`deploy/k8s`). Terraform (`deploy/terraform`) is optional.
- **Rollback.** Code rollback redeploys the previous image tag. Embedding-run rollback is a pointer swap in `artifacts/embeddings/current`. Index rollback is independent (`make index-rollback`, M2).
- **Dependencies.** Dependabot is configured in `.github/dependabot.yml` *(planned)*.

## Security

- Authentication via API key (`X-API-Key`) or JWT; privileged operations require an admin scope.
- Redis-backed rate limiting per key, degrading to a per-instance limiter if Redis is down.
- Strict Pydantic validation: bounded `k`, bounded list sizes, and an allowlist of filter fields.
- Secrets only through environment variables; least-privilege database user.
- Trivy scans in CI and automated dependency updates through Dependabot.
- Logs contain no raw PII.

## Development and testing

Requires Python 3.11+ and GNU Make. Docker is optional and only needed for the full local stack.

```bash
make install-dev          # pip install -e ".[dev]" — the full local environment
make install-hooks        # pre-commit install --install-hooks
make check                # lint + typecheck + check-markers + unit tests
```

Two extras are provided so CI and local development do not pay for what they
do not use:

| Extra | Contents | When to use |
|---|---|---|
| `[dev-lite]` | API + observability + test/quality tooling. **No torch.** | CI lint and unit jobs; fast local iteration. |
| `[inference]` | numpy + onnxruntime + tokenizers. **No torch.** | Serving path, and anything that loads a `.onnx` artifact. |
| `[export]` | sentence-transformers + onnx. Pulls torch. | Producing a new ONNX artifact; the encoder parity test. |
| `[pipeline]` | `[inference]` + pyarrow. | The batch/incremental embedding pipeline and artifact I/O. |
| `[dev]` | Superset of `[dev-lite]` plus embeddings, experiments, churn, bench, and load. | Full local development. |

| Command | What it does |
|---|---|
| `make up` / `make down` | Start / stop the full local stack (Docker Compose) *(planned until M3)* |
| `make fmt` | Ruff auto-fix and format |
| `make lint` | Ruff check and format-check, no modifications |
| `make typecheck` | mypy in strict mode |
| `make check-markers` | Enforce test-tier discipline (unit tests must not require external services) |
| `make test` | Unit + integration tests (integration skips without `RECSYS_TEST_*` env) |
| `make test-unit` | Unit tests only |
| `make test-integration` | Integration tests, light tier (PostgreSQL / Redis); skips encoder parity |
| `make test-encoder` | Encoder parity and the three-tier determinism contract. Requires `[inference,export,pipeline]` |
| `make coverage` | Unit tests with coverage report |
| `make clean` | Remove caches and build artifacts |
| `make migrate` | Apply Alembic migrations *(planned)* |
| `make seed` | Load the sample catalog; `make seed-synthetic N=100000` generates one *(planned)* |
| `make index-promote VERSION=<v>` | Switch the active index; `make index-rollback` reverts *(planned)* |
| `make eval` | Offline evaluation with threshold gate *(planned)* |
| `make load-test` | Locust load test against the local stack *(planned)* |
| `make ab-simulate` | Simulated A/A and A/B experiments with known effects *(planned)* |

Test layers:

- `tests/unit` covers pure logic: preprocessing rules, encoder protocol conformance, artifact primitives (locking, atomic writes, config hashing, run-id allocation, state validation, Parquet I/O), catalog loading, mode planning, assignment hashing, statistics, re-ranking, and fallback selection.
- `tests/integration` runs the API against real PostgreSQL (pgvector) and Redis, plus the encoder tier: ONNX-vs-reference parity, and the full three-tier determinism contract (`tests/integration/test_determinism_tiers.py`). These skip cleanly unless the relevant env vars or extras are present.
- `tests/load` holds the Locust (or k6) scenarios for the latency target.

Pull requests must pass CI. Architectural changes need an ADR in `docs/adr/` — see the [ADR index](docs/adr/README.md) for the convention and the four accepted records (pgvector as default, ONNX Runtime for inference, plain Python CLI for the pipeline, versioned runs with an atomic current pointer).

## Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| pgvector as the default vector store | One datastore for vectors and metadata, transactional filtering, simple operations | Lower throughput ceiling and fewer tuning options than dedicated engines; the FAISS benchmark quantifies the gap |
| HNSW over IVFFlat | Better recall/latency trade-off and no training step | Higher memory use and slower index builds |
| ONNX Runtime for encoding | Faster CPU inference and a smaller runtime footprint | Extra export step; parity with the original model is verified in tests |
| Plain Python CLI for the pipeline | Debuggable anywhere, stable integration surface, no scheduler to install | No retries or scheduling until M6, when an orchestrator wraps the CLI |
| Versioned runs with an atomic `current` pointer | A crash cannot leave a state file pointing at a missing artifact; rollback is a pointer swap | Old runs accumulate on disk until a prune target is added |
| Redis cache keyed by index version and variant | Lower p95 for repeated queries; swaps and A/B arms never serve stale or mixed results | Staleness within the TTL; one more moving part (bypassed on failure) |
| Fallback instead of failing | Availability over freshness | Lower relevance while degraded, tracked through fallback rate and guardrails |
| Hash-based experiment assignment | Stateless, reproducible, consistent across instances | No dynamic re-allocation without re-bucketing users |
| Blue/green indexes | Zero downtime and instant rollback | Roughly double the index storage during a swap |
| Offline evaluation as a CI gate | Catches regressions before deploy | Offline metrics do not guarantee online lift, which is why A/B testing exists |

For the full reasoning behind these choices — including the alternatives that
were considered and rejected — see the ADRs under [`docs/adr/`](docs/adr/).

## Milestones

| # | Milestone | Scope | Exit criteria | Status |
|---|---|---|---|---|
| M0 | Foundation | Repo, CI (lint, type check, test), Docker, pre-commit, first ADR | CI is green on the scaffold; the `docker-build` job builds the image and serves `/healthz`, `/readyz`, and `/metrics` in a container | Done |
| M1 | Embedding pipeline | Batch and incremental embedding, model/index versioning, golden set | Re-running the pipeline produces identical results | Done |
| M2 | Retrieval and offline evaluation | pgvector HNSW, benchmark vs FAISS, Recall@k / NDCG / MRR | Metrics are documented and enforced as a CI gate | Planned |
| M3 | Production API | Auth, rate limiting, caching, fallback, health checks | Integration tests are green | Planned |
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

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).
