# Design Decisions (Fase 0)

> **Status:** locked. Any change to a decision in this file requires a new ADR in `docs/adr/`.

This document captures the pre-flight decisions required before writing feature code. It exists so that later choices (schemas, contracts, refactors) have a single, reviewable source of truth. See [`README.md`](../README.md) for the user-facing description.

---

## 1. Problem statement (one sentence)

**Serve embedding-based item recommendations with low-latency ANN retrieval, honest offline-to-online evaluation, statistically valid A/B testing, and graceful degradation, backed by versioned models and indexes.**

If this sentence cannot be held in one breath, the project has drifted.

---

## 2. Payload vs pipeline

Not everything in the repo carries equal weight. Confusing the two leads to over-engineering the scaffolding and under-engineering the parts reviewers actually look at.

| Category | Items | Why it matters |
|---|---|---|
| **Payload** (the portfolio value) | ANN retrieval quality and latency; re-ranking; fallback behavior; offline evaluation; A/B validity; observability; churn model | These are the claims a reviewer will probe. They must be measured, not asserted. |
| **Pipeline** (supporting) | Embedding job; index build and swap; migrations; Docker; CI/CD; rate limiting; auth | Necessary but generic. Keep them boring, conventional, and out of the way. |

**Rule:** when time is short, cut pipeline; never cut measurement of payload.

---

## 3. Deterministic vs non-deterministic

Determinism is a feature we exploit for idempotency and reproducibility. The table below is the contract.

| Operation | Deterministic? | Idempotency key | Strategy |
|---|---|---|---|
| Embed an item | Yes, given (`model_version`, `content_hash`, preprocessing, batch/thread config) | `sha256(model_version + content_hash)` | Skip re-embed if row exists with same key |
| Build index | Yes, given (embeddings set, HNSW params, item ordering) | `sha256(index_backend + params + catalog_snapshot + model_version)` | New `index_version` written next to live one; never overwrite |
| A/B assignment | Yes | `sha256(f"{experiment_salt}:{user_id}") % 10_000` | Stateless, reproducible across instances; independent across experiments via per-experiment salt |
| Cache entry | Yes, given key | `sha256(index_version + variant + filters + seed_items + k)` | Bounded TTL; safe to recompute on miss |
| Ingest event (`impression`, `click`, `conversion`) | No (external input) | `event_id` (client-supplied UUIDv4) | `INSERT ... ON CONFLICT (event_id) DO NOTHING`; dedup window enforced by uniqueness |
| ANN query result ordering | Partially — ties in score are non-deterministic | n/a | **Tie-break by `item_id` ascending** to make response stable and testable |

**Rule:** if a step is deterministic, its key is a hash of inputs. If it is not, it needs a UUID plus a dedup constraint.

---

## 4. Storage tier

Chosen before any schema is written. Moving tiers later is expensive.

| Tier | Technology | Holds | Why |
|---|---|---|---|
| **Relational + vector** | PostgreSQL 16 + pgvector (HNSW) | Catalog, embeddings, indexes, experiment assignments, event log, churn features | One datastore for vectors *and* metadata means transactional filtering without cross-store consistency problems. Simple to operate on a single node. |
| **Cache / light feature store** | Redis 7 | Response cache keyed by `index_version + variant + query`; short-window counters for rate limiting and feature lookups | Sub-millisecond reads; the cache is **optional** — the service must work (slower) if Redis is down. |
| **Artifacts** | Git LFS or DVC remote (object storage) | Golden set, ONNX model files, index snapshots, offline eval reports | Versioned like code; not queried at runtime. |
| **Not used** | Dedicated vector DB (Milvus/Qdrant/Pinecone); analytics warehouse (BigQuery/Snowflake); object store at request time | — | Deliberately out of scope. If the catalog outgrows single-node pgvector, FAISS benchmark quantifies the gap; migration is a future ADR, not a current concern. |

**Rule:** the request path reads from PostgreSQL and (optionally) Redis only. Nothing else is allowed on the hot path.

---

## 5. Ingestion mode

Two modes, both deliberate. Choosing "both" means we must know exactly when each is used.

| Mode | Trigger | Scope | Where it runs |
|---|---|---|---|
| **Batch** | Model upgrade, catalog re-import, scheduled weekly | Whole catalog | `python -m recsys.embeddings.pipeline --mode=batch`; writes a new `model_version` + `catalog_snapshot` artifact set |
| **Incremental** | Item insert or update (content hash changed) | New or changed items | `python -m recsys.embeddings.pipeline --mode=incremental`; produces an updated artifact set alongside the previous one |

Orchestration (Prefect, Airflow, or a cron wrapper) is deferred to M6 and will
wrap the CLI rather than replace it — see
[`docs/adr/0003-plain-python-cli-for-embedding-pipeline.md`](adr/0003-plain-python-cli-for-embedding-pipeline.md).
The artifact formats and the determinism contract are in
[`docs/embedding-pipeline.md`](embedding-pipeline.md).

**Not used:** streaming (Kafka/Kinesis). Events are appended via `POST /v1/events` into PostgreSQL; they are not consumed as a stream. Adding a stream later is a new ADR.

**Rule:** every embed run records `(model_version, catalog_snapshot, index_version)` in the registry. Re-running the same inputs must produce byte-identical embeddings (checksum verified in CI).

---

## 6. Failure modes

Filled in before writing the API. Every row has an owning mitigation and a metric that makes it visible.

| Failure | Blast radius | Mitigation | Observable via |
|---|---|---|---|
| PostgreSQL unreachable | Retrieval + event logging + experiment assignment fail | Circuit breaker opens; serve popularity fallback within filters; `/readyz` reports **not ready** if DB is required for correctness | `recsys_fallback_total{reason="db_down"}`, `/readyz` |
| pgvector index not loaded on an instance | That instance cannot serve ANN | `/readyz` fails → load balancer drains it; other instances keep serving | `/readyz`, rollout alarms |
| ANN query exceeds `ANN_TIMEOUT_MS` | One request degrades | Cancel query; serve popularity fallback; increment counter | `recsys_fallback_total{reason="ann_timeout"}` |
| Redis unavailable | Cache misses; rate limiter loses shared state | Cache is bypassed (not failed); rate limiter degrades to per-instance in-memory limiter | `recsys_cache_requests_total{result="bypass"}`, rate-limit-degraded log |
| ONNX runtime OOM / model file missing | Embedding + query encoding unavailable on that instance | `/readyz` fails; instance drained; no partial responses | `/readyz`, startup validation |
| Upstream auth provider (JWT) unreachable | API keys still work; JWT auth fails | API-key path unaffected; JWT requests get `503` | `recsys_errors_total{type="auth_upstream"}` |
| Disk full (Postgres WAL) | Writes fail; reads may still succeed briefly | Alert on `pg_database_size_bytes`; runbook step: vacuum, archive WAL, or promote standby | Grafana alert `disk_full` |
| GitHub Actions outage | No deploys; no CI | Manual `git push` still works; no runtime impact | n/a (developer-visible) |
| Index swap goes wrong | All instances serve a bad index | Blue/green swap is a single pointer update; rollback is `make index-rollback` (< 5 s) | `recsys_active_index_info`, eval gate before promote |

**Rule:** every failure row must answer three questions — *what breaks, how it degrades, how we see it*.

---

## 7. Out of scope (recorded so we do not drift)

- Multi-tenancy, admin UI, billing.
- Model fine-tuning (only the churn model is trained here).
- Real-time streaming ingestion.
- Multi-region replication, sharding, 10M+ item catalogs.
- RAG or LLM text generation — embeddings are used for item recommendation, not generation.

Anything in this list that becomes necessary is a new ADR, not a quiet addition.

---

## 8. Summary checklist

| Pre-flight question | Answer |
|---|---|
| Problem in 1 sentence | ✅ Section 1 |
| Payload vs pipeline | ✅ Section 2 |
| Deterministic vs non-deterministic | ✅ Section 3 |
| Storage tier | ✅ Section 4 |
| Ingestion mode | ✅ Section 5 |
| Failure modes | ✅ Section 6 |
| Out of scope | ✅ Section 7 |

With this document merged, feature work may begin. The next milestone (M0.3) adds the repository scaffolding: `.gitignore`, `LICENSE`, `pyproject.toml`, `.env.example`, `CHANGELOG.md`.
