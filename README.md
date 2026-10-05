# embedding-recommender-api

[

![CI](https://github.com/WoodinGlass/embedding-recommender-api/actions/workflows/ci.yml/badge.svg)

](https://github.com/WoodinGlass/embedding-recommender-api/actions/workflows/ci.yml)


![Python 3.11](https://img.shields.io/badge/python-3.11-blue)




![License: MIT](https://img.shields.io/badge/license-MIT-green)



Embedding-based recommendation service with low-latency ANN retrieval (target p95 < 200 ms), re-ranking, statistically valid A/B testing, monitoring, and automated deployment (Docker + CI/CD). It ships with offline evaluation, model/index versioning, fallbacks, and a churn-risk extension.

> **Status:** in development. Progress is tracked in [Milestones](#milestones). Performance figures are targets until the M4 benchmark is published.

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

1. **Embedding pipeline** — batch and incremental catalog embedding with versioned models and indexes.
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
  subgraph OFFLINE["Offline / batch (Prefect or Airflow)"]
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
| API | Python 3.11, FastAPI, Pydantic v2 |
| Embeddings | sentence-transformers, exported to ONNX Runtime for fast CPU inference |
| Vector search | pgvector (HNSW) by default; FAISS as a benchmark option |
| Data and cache | PostgreSQL; Redis (cache and lightweight feature store) |
| Pipelines and versioning | Prefect or Airflow (batch re-embed); DVC or MLflow (model registry) |
| Observability | Prometheus, Grafana, OpenTelemetry, structlog |
| Quality | pytest, Ruff, mypy, pre-commit, Locust or k6 |
| Infrastructure | Docker (multi-stage), docker-compose (dev), GitHub Actions, Terraform or Helm (optional) |
| Security | API key / JWT, rate limiting, Dependabot, Trivy |

## Project structure

```text
embedding-recommender-api/
├── src/recsys/
│   ├── api/                # routers, schemas, middleware (auth, rate limit)
│   ├── embeddings/         # encoder, batch pipeline, ONNX export
│   ├── retrieval/          # ANN index, filtering, re-ranker
│   ├── fallback/           # popularity and cold-start
│   ├── experiments/        # assignment, logging, stats analysis
│   ├── churn/              # features, model, scoring
│   ├── monitoring/         # metrics, drift detection
│   └── config/             # settings per environment
├── pipelines/              # batch embed and index rebuild (flows/DAGs)
├── evaluation/             # offline eval (Recall@k, NDCG, MRR), golden set
├── migrations/             # Alembic
├── data/                   # samples only; real data via DVC
├── tests/
│   ├── unit/
│   ├── integration/
│   └── load/
├── deploy/
│   ├── docker/
│   ├── k8s/
│   └── terraform/
├── dashboards/             # Grafana JSON
├── docs/
│   ├── adr/                # architecture decision records, e.g. 0001-pgvector-default.md
│   ├── runbook.md
│   └── api.md
├── .github/
│   ├── workflows/          # ci.yml, cd.yml, security.yml
│   └── dependabot.yml
├── .env.example
├── .pre-commit-config.yaml
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── LICENSE
```

## Quickstart

Prerequisites: Docker with Compose v2, GNU Make, and Python 3.11 (for local development).

```bash
git clone https://github.com/<your-username>/embedding-recommender-api.git
cd embedding-recommender-api

cp .env.example .env       # set API keys and database credentials
make up                    # API, PostgreSQL (pgvector), Redis, Prometheus, Grafana
make migrate               # apply Alembic migrations
make seed                  # load the sample catalog from data/sample/
make embed                 # embed the catalog, build the first index, and activate it
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

Settings are read from environment variables (see `.env.example`). Never commit real secrets; `.env` is git-ignored.

| Variable | Description | Example |
|---|---|---|
| `APP_ENV` | Selects the settings profile in `src/recsys/config/` | `dev` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://recsys:recsys@postgres:5432/recsys` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `API_KEYS` | Comma-separated API keys (or use JWT instead) | `dev-key-1` |
| `JWT_SECRET` | Signing secret when JWT auth is enabled | — |
| `RATE_LIMIT_PER_MINUTE` | Per-key request limit | `600` |
| `EMBEDDING_MODEL` | sentence-transformers model exported to ONNX | `sentence-transformers/all-MiniLM-L6-v2` |
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
    "model_version": "minilm-onnx-v1",
    "index_version": "idx-0003",
    "experiment": { "name": "rerank_mmr", "variant": "treatment" }
  }
}
```

Errors: `401`/`403` for auth, `422` for validation, `429` with `Retry-After` when rate limited, and `503` only when both ANN retrieval and the fallback fail. Interactive OpenAPI docs are served at `/docs`; a summary lives in `docs/api.md`.

## Embedding pipeline and index lifecycle

- **Batch and incremental runs.** Every item carries a content hash. Incremental runs embed only new or changed items; batch runs re-embed the whole catalog, for example after a model upgrade.
- **Versioning.** Each run records `model_version`, `catalog_snapshot`, and `index_version` in the registry (DVC or MLflow). Embeddings and indexes are written as new versions, never overwritten in place.
- **Deterministic by design.** Pinned model version, fixed text preprocessing, stable item ordering, and fixed batch/thread settings: the same catalog snapshot and model version produce the same embeddings (checksum-verified).
- **Blue/green index swap.** A new index version is built next to the live one, evaluated against the golden set, warmed up, and then promoted by switching the active-version pointer. API instances pick up the new pointer without a restart, and the previous version is kept for instant rollback.

```bash
make index-promote VERSION=idx-0004
make index-rollback
```

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

- **Golden set.** Query-to-relevant-items pairs in `evaluation/`, versioned with DVC.
- **Retrieval quality.** Recall@k, NDCG@k, and MRR against the golden labels.
- **ANN fidelity.** Overlap with exact (brute-force) kNN, so index tuning is not mistaken for a relevance change.
- **CI gate.** `make eval` writes a JSON report and fails if any metric drops below `evaluation/thresholds.yaml`. CI runs it on a small fixture set to stay fast; the full golden set runs before every index promotion.

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
- **Proof it works.** `make ab-simulate` replays synthetic traffic with known effects. A/A runs should produce false positives at about the chosen significance level, and A/B runs with an injected lift should be detected at the planned power.

```bash
python -m recsys.experiments.analyze --experiment rerank_mmr
```

## Observability

- **Metrics** (Prometheus, `/metrics`): `recsys_request_duration_seconds` (histogram by route, source, and status), `recsys_cache_requests_total{result}`, `recsys_fallback_total{reason}`, `recsys_errors_total{type}`, `recsys_embedding_drift_score`, and `recsys_active_index_info{index_version,model_version}`. Histogram buckets include 0.2 s so the latency target is directly measurable.
- **p95 query:** `histogram_quantile(0.95, sum by (le) (rate(recsys_request_duration_seconds_bucket{route="/v1/recommend"}[5m])))`
- **Tracing and logs.** OpenTelemetry spans around cache, ANN query, re-rank, and fallback. structlog JSON logs carry `request_id` and `trace_id`, with no raw PII.
- **Drift.** Recent query and item embeddings are compared with a reference window using centroid cosine shift and PSI over the top PCA components.
- **Alerts.** p95 > 200 ms for 10 minutes; 5xx rate > 1% for 5 minutes; sustained fallback-rate spike; sharp drop in cache hit rate; drift score above threshold; SRM detected in a running experiment.
- **Dashboards.** Grafana JSON in `dashboards/` (service health, cache and fallback, drift, experiments).

## Performance targets and results

**Target:** p95 < 200 ms on 100k+ items at the target RPS (fixed in M4).

**Method:** Locust scenarios in `tests/load/`. Latency comes from the server-side Prometheus histogram and is cross-checked against client-side percentiles. Warm-cache and cold-cache runs are reported separately, and every result records hardware, dataset size, and commit SHA.

```bash
make seed-synthetic N=100000
make embed
make load-test
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
- **Monitoring.** Feature and score drift appear on the same dashboards as embedding drift.

## Deployment and CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Pull request, push to `main` | Ruff, mypy, unit and integration tests (PostgreSQL with pgvector and Redis service containers), offline evaluation gate, Docker build |
| `cd.yml` | Push to `main` | Build and push the image (tagged with the git SHA), deploy to staging, run smoke tests and the eval gate, promote to production after approval, auto-rollback if readiness or SLO checks fail |
| `security.yml` | Pull request, nightly | Trivy scans (filesystem and image), dependency review |

- **Image.** Multi-stage, non-root Docker image. `docker-compose.yml` is for local development only.
- **Orchestration.** Rolling updates with readiness probes (`deploy/k8s`). Terraform (`deploy/terraform`) is optional.
- **Rollback.** Code rollback redeploys the previous image tag. Index rollback is independent (`make index-rollback`).
- **Dependencies.** Dependabot is configured in `.github/dependabot.yml`.

## Security

- Authentication via API key (`X-API-Key`) or JWT; privileged operations require an admin scope.
- Redis-backed rate limiting per key, degrading to a per-instance limiter if Redis is down.
- Strict Pydantic validation: bounded `k`, bounded list sizes, and an allowlist of filter fields.
- Secrets only through environment variables; least-privilege database user.
- Trivy scans in CI and automated dependency updates through Dependabot.
- Logs contain no raw PII.

## Development and testing

```bash
pip install -e ".[dev]"   # dev dependencies from pyproject.toml
pre-commit install        # Ruff and mypy run on every commit
```

| Command | What it does |
|---|---|
| `make up` / `make down` | Start / stop the full local stack |
| `make migrate` | Apply Alembic migrations |
| `make seed` | Load the sample catalog (`make seed-synthetic N=100000` generates a synthetic one) |
| `make embed` | Run the embedding pipeline and build a new index version |
| `make index-promote VERSION=<v>` | Switch the active index; `make index-rollback` reverts |
| `make eval` | Offline evaluation with threshold gate |
| `make lint` / `make typecheck` | Ruff / mypy |
| `make test` | Unit and integration tests |
| `make load-test` | Locust load test against the local stack |
| `make ab-simulate` | Simulated A/A and A/B experiments with known effects |

Test layers:

- `tests/unit` covers pure logic: assignment hashing, statistics, re-ranking, and fallback selection.
- `tests/integration` runs the API against real PostgreSQL (pgvector) and Redis, including failure paths such as Redis down or index unavailable.
- `tests/load` holds the Locust (or k6) scenarios for the latency target.

Pull requests must pass CI. Architectural changes need an ADR in `docs/adr/`.

## Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| pgvector as the default vector store | One datastore for vectors and metadata, transactional filtering, simple operations | Lower throughput ceiling and fewer tuning options than dedicated engines; the FAISS benchmark quantifies the gap |
| HNSW over IVFFlat | Better recall/latency trade-off and no training step | Higher memory use and slower index builds |
| ONNX Runtime for encoding | Faster CPU inference and a smaller runtime footprint | Extra export step; parity with the original model is verified in tests |
| Redis cache keyed by index version and variant | Lower p95 for repeated queries; swaps and A/B arms never serve stale or mixed results | Staleness within the TTL; one more moving part (bypassed on failure) |
| Fallback instead of failing | Availability over freshness | Lower relevance while degraded, tracked through fallback rate and guardrails |
| Hash-based experiment assignment | Stateless, reproducible, consistent across instances | No dynamic re-allocation without re-bucketing users |
| Blue/green indexes | Zero downtime and instant rollback | Roughly double the index storage during a swap |
| Offline evaluation as a CI gate | Catches regressions before deploy | Offline metrics do not guarantee online lift, which is why A/B testing exists |

## Milestones

| # | Milestone | Scope | Exit criteria | Status |
|---|---|---|---|---|
| M0 | Foundation | Repo, CI (lint, type check, test), Docker, pre-commit, first ADR | CI is green on the scaffold and `make up` serves `/healthz` | Planned |
| M1 | Embedding pipeline | Batch and incremental embedding, model/index versioning, golden set | Re-running the pipeline produces identical results | Planned |
| M2 | Retrieval and offline evaluation | pgvector HNSW, benchmark vs FAISS, Recall@k / NDCG / MRR | Metrics are documented and enforced as a CI gate | Planned |
| M3 | Production API | Auth, rate limiting, caching, fallback, health checks | Integration tests are green | Planned |
| M4 | Observability and load test | Prometheus/Grafana, tracing, Locust | p95 < 200 ms at the target RPS, with evidence committed in `docs/` | Planned |
| M5 | A/B testing | Assignment, logging, statistical analysis, dashboard | Simulation reaches the correct conclusion on a known effect | Planned |
| M6 | Deployment | Automated CD, staging to prod, rollback, blue/green index | A merge to `main` reaches staging automatically; production promotion, rollback, and index swap are demonstrated with no downtime | Planned |
| M7 | Churn extension | Features, model, endpoint, drift monitoring | Endpoint serves scores; AUC and calibration documented; drift monitored | Planned |
| M8 | Hardening | Security scan, runbook, simulated postmortem, final README | No unaddressed high/critical findings; runbook and postmortem published; README documents architecture and trade-offs | Planned |

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE).
