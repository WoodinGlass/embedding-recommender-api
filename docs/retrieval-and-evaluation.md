# Retrieval and offline evaluation

Design doc for M2. Decisions are recorded as ADRs 0005–0012; this document
is the consolidated view — what is being built, how the pieces fit, and
where the boundaries are. When this document and an ADR disagree, the ADR
wins, and the disagreement is a bug to be fixed in the same change.

> **Status:** locked for M2. Changes to artifact formats, metric
> definitions, or the index identity require a new ADR.

---

## 1. Scope

**In scope for M2**

- A `Backend` protocol and two implementations: `PgvectorBackend`
  (production) and `NumpyBackend` (exact kNN, Colab and baseline).
- The pgvector schema (`item`, `embedding`, `index_registry`) and its
  Alembic migration.
- `index_version` identity and blue/green lifecycle: `make index-build`,
  `make index-promote`, `make index-rollback`.
- A versioned golden set, metric definitions, three baselines, and an
  evaluation runner that writes a report.
- Thresholds enforced as a CI gate (`make eval`), with an absolute floor,
  a history file, and staleness checks.
- A FAISS benchmark methodology and script, producing a JSON and a
  generated Markdown summary.

**Out of scope for M2**

- Re-ranking (popularity, recency, diversity/MMR). Ships in M3 as an
  experiment arm (ADR-0005).
- Serving over HTTP (`/v1/recommend` still returns `503`). Ships in M3.
- Caching, rate limiting, auth. Ships in M3.
- Load testing and the p95 latency target. Ships in M4.
- Real popularity (from an event log). The M2 baseline is synthetic;
  the interface is ready for M5.
- Partial indexes for extreme filter selectivity. Documented as a
  future ADR in ADR-0008.

---

## 2. Module layout

```
src/recsys/retrieval/
├── __init__.py
├── base.py              # IndexBackend protocol (ADR-0012)
├── registry.py          # INDEX_BACKEND enum → factory (ADR-0012)
├── filters.py           # FILTER_FIELDS allowlist (ADR-0008)
├── identity.py          # index_version hash (ADR-0007)
├── pgvector.py          # PgvectorBackend (ADR-0006, ADR-0008, ADR-0012)
├── numpy_backend.py     # NumpyBackend (ADR-0012)
└── rerank.py            # Reranker protocol stub (ADR-0005; empty in M2)

src/recsys/evaluation/
├── __init__.py
├── metrics.py           # recall_at_k, ndcg_at_k, mrr (ADR-0009)
├── golden_set.py        # loader + validator for v1.jsonl (ADR-0009)
├── baselines.py         # random, synthetic popularity, exact kNN (ADR-0009)
├── runner.py            # orchestration + report writer (ADR-0010)
└── thresholds.py        # loader + gate logic (ADR-0010)

scripts/
├── build_index.py       # make index-build (ADR-0006, ADR-0007)
├── promote_index.py     # make index-promote, uses pg_advisory_lock
├── rollback_index.py    # make index-rollback
├── bench_faiss.py       # FAISS benchmark (ADR-0011)
└── render_benchmark.py  # JSON → Markdown (ADR-0011)

migrations/versions/
└── 0001_pgvector_schema.py   # Alembic (ADR-0006)

evaluation/
├── golden_set/v1.jsonl       # golden set, committed (ADR-0009)
├── thresholds.yaml           # gate thresholds (ADR-0010)
├── thresholds_history.yaml   # audit trail (ADR-0010)
└── report.json               # generated, git-ignored
```

Each module has one responsibility. `metrics.py` is pure functions with no
I/O; `runner.py` does I/O and calls `metrics.py`. This mirrors the M1 split
between `preprocess.py` (pure) and `pipeline.py` (I/O).

---

## 3. Backend protocol

The interface every retrieval backend satisfies is defined in
`src/recsys/retrieval/base.py`. Its rationale (why synchronous, why
`NDArray[np.float32]`, why the tie-break lives in the caller) is in
ADR-0012. Summary:

```python
class IndexBackend(Protocol):
    name: str

    def is_ready(self) -> bool: ...

    def search(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]: ...
```

`PgvectorBackend` is the production path. `NumpyBackend` is for exact
search — Colab end-to-end runs and the ANN-fidelity baseline. FAISS is
not a backend; it runs inside the benchmark script (ADR-0011).

Both implementations agree on top-k results for exact search, verified by
a conditional integration test (runs in CI, skips in Colab). The
tie-breaking rule (item_id ascending on equal scores) is applied by the
caller, in `runner.py` and — from M3 — in the recommend endpoint.

---

## 4. Schema

Full DDL and rationale in ADR-0006. Summary:

- `item` — catalog rows, with `content_hash`, `language`, `created_at`,
  `updated_at`. Indexed on `category`, `brand`, `language` for filter
  selectivity.
- `embedding` — one row per `(item_id, index_version)`, with
  `vector(384)`. HNSW index with `m=16, ef_construction=64` (values from
  config; changing them changes `index_version`).
- `index_registry` — one row per `index_version`, recording every
  parameter that affects the index's behavior: `model_version`,
  `catalog_snapshot`, `preprocessing_version`, `metric`, `hnsw_m`,
  `hnsw_ef_construction`, `hnsw_ef_search`, `pgvector_version`,
  `golden_set_version`, `row_count`, `status`, timestamps.

At most one `active` index at any moment, enforced by a partial unique
index. `status` transitions: `building` → `active` → `retired`. Nothing
else.

The `vector(384)` dimension is fixed. A model swap requires a new
migration that drops and recreates the column; the procedure is in
ADR-0006 § Model swap procedure.

---

## 5. Index identity

`index_version = "idx-" + sha256(canonical_json)[:8]`.

The canonical JSON contains every parameter that affects which items a
query returns: `model_version`, `catalog_snapshot`,
`preprocessing_version`, `metric`, `hnsw_m`, `hnsw_ef_construction`,
`pgvector_version`. Deliberately excluded: `hnsw_ef_search` (query-time
knob) and `golden_set_version` (evaluation metadata).

Collision handling: an 8-hex prefix (2^32) is checked against
`index_registry`. If a row exists with **different** hash inputs, the
prefix extends to 12 hex. If a row exists with **identical** inputs, the
build is treated as a duplicate and `make index-build` is a no-op.

`pgvector_version` is captured at build time (`SELECT extversion FROM
pg_extension`) and stored in the registry. It is not re-queried at runtime
— the identity describes the index as built, not the cluster as
configured. Rationale in ADR-0007.

---

## 6. Filter strategy

Full rationale in ADR-0008. Summary:

**Version detection.** On the first connection that uses the retrieval
layer, query the pgvector extension version once. Log it at `INFO`. If
below 0.8, log one `WARNING`. Do not re-query per request.

**With iterative scan (pgvector >= 0.8).** `SET LOCAL
hnsw.iterative_scan = 'strict_order'` and `SET LOCAL
hnsw.max_scan_tuples = 20000` on filtered queries. `strict_order` is
chosen because the API contract guarantees sorted results with a
deterministic tie-break; `relaxed_order` would require a re-sort and
complicate tie-breaking.

**Without iterative scan (pgvector < 0.8).** Over-fetch by a factor of
four (`effective_k = k * 4`), post-filter in Python, and log a `WARNING`
per invocation with the reason. The multiplier is a constant in code, not
a config knob — moving the decision into config would invite drift
between environments and make reports harder to compare.

**Filter allowlist.** The set of filter fields (`category`, `brand`,
`language`) is defined in `docs/contracts.md` § 2.1 and read by the code
from a single `FILTER_FIELDS` constant in `filters.py`. Filter values
are always SQL parameters, never string-interpolated.

**Known limitation.** The `k * 4` fallback is correct for typical
selectivity (10–30%). It is inadequate for extreme selectivity (< 1%),
where the correct answer is a partial index or a two-stage query. That
work is deferred; the ADR documents the gap.

---

## 7. Golden set, metrics, and baselines

Full rationale in ADR-0009. Summary:

**Golden set.** `evaluation/golden_set/v1.jsonl`, one JSON object per
query: `query_id`, `topic`, `seed_item_ids`, `relevant_item_ids`. Versioned
by filename (`v1.jsonl`, `v2.jsonl`). The version string `"v1"` is declared
in `thresholds.yaml` and recorded in `index_registry` and every report. A
report is comparable to another only when the golden set versions match.

**Metrics.** Pure functions in `metrics.py`:

- `recall_at_k(retrieved, relevant, k) = |retrieved[:k] ∩ relevant| / min(|relevant|, k)`
- `ndcg_at_k(retrieved, relevant, k)` with binary relevance and the standard
  `DCG/IDCG` formula.
- `mrr(retrieved, relevant) = 1 / rank(first relevant)`, or 0.

Aggregated as the arithmetic mean over queries. Per-topic breakdown is
recorded for diagnostics.

**Query encoding.** `normalize(mean(seed embeddings))`. Deterministic,
parameter-free, and appropriate for a golden set whose seeds within a
query share a topic. Alternatives (max-pool, weighted, concatenation)
are recorded in ADR-0009 as rejected for M2.

**Baselines.** Three, all evaluated with the same metrics and the same
golden set:

- **Random** — deterministic per `query_id` (seed = `sha256(query_id)`).
- **Popularity (synthetic)** — `sha256(item_id) % 1000`, sorted descending,
  ties by `item_id`. This is a placeholder; the interface
  (`PopularityProvider.scores_for`) is ready for M5's real event-based
  popularity.
- **Exact kNN** — `NumpyBackend` over the same vectors. The ceiling for
  recall; the reference for ANN fidelity.

**ANN fidelity.** `|top_k(HNSW) ∩ top_k(exact)| / k`, per query, averaged.
This isolates "the index lost neighbors" from "the vectors changed" —
two failure modes that Recall@k alone cannot distinguish.

---

## 8. Evaluation gate

Full rationale in ADR-0010. Summary:

**Thresholds.** `evaluation/thresholds.yaml` declares an `absolute_floor`
per metric and a per-system threshold. The gate checks both. The
absolute floor prevents a slow drift from laundering regressions into the
per-system thresholds.

**Setting thresholds.** Measured value minus a margin, chosen by a human
after reviewing the measurement. Not auto-computed: a measured value can
itself be a regression, and only a human who reads the diff can tell
"the metric dropped because the code is correct" from "the metric dropped
because something is wrong".

**History.** `evaluation/thresholds_history.yaml` records every threshold
change with the measured values at the time. A lowered threshold requires
a `reason` field. The gate does not enforce the presence of a reason —
that is a code review matter — but the file makes the absence visible.

**Report checks.** Before evaluating any threshold, the gate verifies:

1. Golden set version matches between the report and the threshold file.
2. The report's commit matches the commit under test.
3. The report is no older than `max_report_age_hours` (default 24).

Any mismatch is a **fail**, not a skip.

**CI job.** A dedicated `evaluation` job (separate from `test-integration`
and `test-encoder`), with `timeout-minutes: 10`. It runs migrations,
builds and promotes an index, runs `make eval`, and uploads
`evaluation/report.json` as a build artifact regardless of exit code.

**Exit codes.** `0` all thresholds met; `1` a threshold failed or a
version/staleness check failed; `2` evaluation could not run (missing
index, unreachable DB). Codes 1 and 2 have different remediation; the
gate distinguishes them.

---

## 9. FAISS benchmark

Full rationale in ADR-0011. Summary:

**What is compared.** On the same vectors, same golden set, same
parameters: exact kNN (numpy), pgvector HNSW, FAISS HNSW.

**Reproducibility.** Every run records commit, `catalog_snapshot`,
`model_version`, `golden_set_version`, `vector_dim`, `vector_count`,
hardware (CPU model, core count, RAM, OS, kernel), library versions
(numpy, faiss, pgvector, PostgreSQL), parameters (`hnsw_m`,
`hnsw_ef_construction`, `ef_search` grid, seed=42, thread count), and
per-query latencies. Output: `docs/faiss-benchmark.json` (raw) and
`docs/faiss-benchmark.md` (generated).

**Latency asymmetry.** Server-side for pgvector (via `EXPLAIN ANALYZE`),
in-process for FAISS and numpy. Documented in the JSON and the Markdown.
The choice is deliberate: the benchmark compares ANN libraries, and a
socket round-trip is not part of either library.

**Not a CI gate.** Latency from a shared runner is a property of the
runner. The benchmark runs on demand, the JSON and generated Markdown
are committed, and the README table is written by hand after review.

---

## 10. Environments

**CI (GitHub Actions).** Every backend, every test. The `test-integration`
job provisions a `pgvector/pgvector:pg16` service container and sets
`RECSYS_TEST_DATABASE_URL`. The `evaluation` job runs the full gate.

**Google Colab.** Development, lint, unit tests, and end-to-end
evaluation via `NumpyBackend`. Does not run pgvector integration tests
(no Docker daemon). Does not run the FAISS benchmark (no `[bench]` extra
by default). The three-tier determinism suite from M1 runs unchanged.

**Local (developer machine with Docker).** Same as CI minus the CI jobs.

The gap is not a bug; it is the same trade-off M1 made with ONNX. Colab
develops the code; CI proves it against the real dependency.

---

## 11. Build, promote, rollback

The lifecycle is a state machine over `index_registry.status`.

**Build** (`make index-build`)

1. Read the active embedding run via `artifacts/embeddings/current`.
2. Compute `index_version` (ADR-0007). If a row exists with identical
   inputs, exit 0 without rebuilding.
3. Insert a `building` row.
4. Insert `item` rows (upsert) and `embedding` rows.
5. Update the row's `row_count` and leave it in `building`.

The `building` state is a staging area. A build that crashes leaves a
`building` row that no reader sees; the next build either reuses it
(idempotent) or starts a new one.

**Promote** (`make index-promote VERSION=<v>`)

1. Take a `pg_advisory_lock` on a fixed key (so two concurrent promotes
   cannot race).
2. Verify the target is in `building` status and its `row_count` matches
   the table.
3. Mark the current `active` row `retired` (with `retired_at`).
4. Mark the target `active` (with `activated_at`).
5. Release the advisory lock.

The partial unique index on `status = 'active'` is the safety net: a
race that somehow bypassed the advisory lock fails at the `UPDATE`, not
silently.

**Rollback** (`make index-rollback`)

Same as promote, with the target being the most recent `retired` row.
The `current` active index is retired; the previous one becomes active
again. This is a pointer swap; the retired rows remain in `embedding` for
instant reuse.

**Prune** is not implemented in M2. Rows accumulate by design; a future
ADR adds `make prune-indexes --keep=N` when accumulation becomes a
problem. The threshold is documented (1M items × 2 indexes ≈ 3 GB) so
the decision is deferred with a number, not with "later".

---

## 12. Observability

M2 wires the metrics M4 will consume; it does not yet emit dashboards.

- **`recsys_active_index_info{index_version, model_version}`** — a gauge
  of 1, updated when the active index changes. `/readyz` and the
  evaluation gate both depend on this being correct.
- **Log lines** from the retrieval layer, all JSON via `structlog`:
  - `retrieval.pgvector.version` (INFO, once at startup)
  - `retrieval.pgvector.old_version` (WARNING, once at startup, if < 0.8)
  - `retrieval.pgvector.fallback` (WARNING, per filtered query on < 0.8)
  - `index.build.done` / `index.build.error`
  - `index.promote.done` / `index.promote.error`
  - `index.rollback.done`
- **Query latency** is measured inside the backend and passed back to
  the caller (which is where M3 will record it in the histogram). The
  backend does not emit the histogram itself; that is the serving layer's
  job. M2 provides the timing; M3 provides the metric.

The `evaluation` job does not depend on Prometheus; it reads the report
file and the threshold file. Prometheus and Grafana are M4.

---

## 13. What lands in M3

- `POST /v1/recommend` and `GET /v1/items/{item_id}/similar` stop
  returning `503`.
- The `Reranker` protocol gets implementations (popularity, recency,
  MMR) and a call site.
- The recommend endpoint composes: cache lookup, backend search, re-rank,
  fallback, response. The cache key includes `index_version` and the
  experiment variant.
- Redis caching and the circuit breaker for the Redis-down path.
- Rate limiting and API-key auth on the routes.
- `/readyz` performs real checks (DB reachable, active index loaded,
  Redis status reported but not required).
- The evaluation table gains a "with re-ranker" row.

None of these change the artifact formats, metric definitions, or index
identity defined here. That stability is the point of writing this
document before M3 starts.

---

## 14. Summary of decisions

| Topic | Decision | ADR |
|---|---|---|
| M2 scope | Retrieval only; re-ranker in M3 | [0005](adr/0005-m2-scope-retrieval-only.md) |
| Schema | Three tables, `vector(384)`, explicit model-swap procedure | [0006](adr/0006-pgvector-schema.md) |
| `index_version` | `idx-<sha8>` over seven inputs, sha12 on collision | [0007](adr/0007-index-version-identity.md) |
| Filter strategy | Startup version detection, `strict_order` iterative scan, `k*4` fallback | [0008](adr/0008-filter-strategy.md) |
| Golden set + metrics | `v1.jsonl`, Recall@k / NDCG@k (binary) / MRR, mean query encoding | [0009](adr/0009-golden-set-and-metrics.md) |
| Thresholds | Absolute floor + per-system threshold + history + staleness checks | [0010](adr/0010-evaluation-thresholds.md) |
| FAISS benchmark | Same vectors, hardware metadata, seed=42, not a CI gate | [0011](adr/0011-faiss-benchmark-methodology.md) |
| Backends | Sync `NDArray` protocol; `NumpyBackend` for exact kNN; pgvector in CI only | [0012](adr/0012-backend-abstraction.md) |

With this document merged, the M2 code may begin.
