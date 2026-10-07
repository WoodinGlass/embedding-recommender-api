# ADR-0011: FAISS benchmark methodology and reproducibility

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

`README.md` § Performance and § Offline evaluation promise a FAISS comparison:
the M2 table has a row for "FAISS HNSW (benchmark)", and ADR-0001 cites the
benchmark as the mechanism that quantifies the gap between pgvector and a
dedicated ANN library. The exit criteria for M2 do not require the benchmark
to gate CI; they require the numbers to exist, to be reproducible, and to be
recorded in the README.

Three problems make a naïve "run FAISS, paste the numbers" approach
misleading:

1. **Apples to oranges.** FAISS and pgvector differ in build parameters,
   metric implementations, and where the latency is measured (in-process vs
   over a socket). A comparison that ignores these differences produces a
   number that says nothing useful.
2. **Non-reproducible numbers.** FAISS's HNSW builder uses randomization.
   Without a fixed seed and a recorded environment, the same script on two
   machines produces different graphs and different recall. The number in
   the README becomes unauditable.
3. **Stale artifacts.** A benchmark writes a table; the table drifts from
   the code that produced it. Without a link from the reported number to
   the commit, hardware, and inputs, the reader cannot tell whether the
   number describes the current system.

The benchmark is a **reported measurement**, not a gate. It runs on demand
or in a scheduled job, publishes a JSON file and a generated Markdown
summary, and the maintainer pastes the numbers into `README.md` after
review. That distinction matters: gate numbers must be cheap and
deterministic; reported numbers can be expensive and depend on hardware.

## Decision

### What is compared

The benchmark compares, on the **same embedding vectors** and the **same
golden set**:

| System | Path measured |
|---|---|
| **Exact kNN (numpy)** | In-process cosine similarity over all vectors. The ceiling for recall; the floor for latency on the sample catalog. |
| **pgvector HNSW** | SQL query over the HNSW index, executed over a local Unix-socket connection. Latency measured server-side. |
| **FAISS HNSW** | In-process `IndexHNSWFlat`, cosine metric. Latency measured around the `search` call. |

The pgvector numbers in the benchmark are *not* the same as the M2 evaluation
numbers. The evaluation numbers come from the same query path the API will
use (over TCP, with auth and middleware in M3), and are the ones that
populate the README evaluation table. The benchmark numbers are a
like-for-like comparison of ANN libraries, isolated from the serving stack.
Both are published, and the difference is explained.

### Reproducibility requirements

Every benchmark run records, in `docs/faiss-benchmark.json`:

**Inputs**

- `commit` — `git rev-parse HEAD` at run time. A benchmark without a commit
  is unverifiable.
- `catalog_snapshot` — from the active run's manifest (M1).
- `model_version` — same source.
- `golden_set_version` — the version of the golden set used.
- `vector_dim` — the embedding dimension, to catch the case where the
  benchmark was run against a different model.
- `vector_count` — the number of vectors indexed.

**Environment**

- `cpu_model` — from `/proc/cpuinfo` on Linux, `sysctl` on macOS.
- `cpu_count` — number of logical processors.
- `ram_bytes` — from `/proc/meminfo` or `sysctl`.
- `os` — `platform.system()` and `platform.release()`.
- `kernel` — `platform.version()`.
- `python_version`.
- `numpy_version`.
- `faiss_version` — `faiss.__version__`.
- `pgvector_version` — from the active index's registry row.
- `postgres_version` — `SELECT version()`.

**Parameters**

- `hnsw_m`, `hnsw_ef_construction` — same values for both libraries,
  read from `index_registry` for pgvector and passed explicitly to FAISS.
- `hnsw_ef_search` — the query-time knob; the benchmark sweeps it over a
  fixed grid `[40, 80, 160, 320]` and reports per-value metrics.
- `seed` — fixed at 42 for FAISS HNSW construction.
- `warmup_queries` — the number of queries discarded before measurement
  (default 50), to exclude cache-cold effects from the reported latency.
- `measured_queries` — the number of queries measured (default 500,
  re-using the golden set queries cyclically if it has fewer).

**Results** (per system, per `ef_search` value)

- `recall_at_10` — against the golden set, defined per ADR-0009.
- `ann_recall_vs_exact` — overlap with exact kNN, defined per ADR-0009.
- `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms` — measured as
  described below.
- `qps` — queries per second, computed as `measured_queries / total_time`.
- `index_build_time_s` — wall clock for building the index.
- `index_memory_bytes` — RSS delta after building the index and before
  querying, on a best-effort basis.
- `index_size_bytes` — for FAISS, the size of the serialized index file;
  for pgvector, `pg_relation_size('embedding_hnsw_cosine_idx')` plus the
  table size.

### Latency measurement

For **numpy exact kNN** and **FAISS**, latency is measured as the wall
clock around the search call in a single-threaded loop:

```python
for _ in range(warmup):
    index.search(q, k)
t0 = time.perf_counter()
for q in queries:
    index.search(q, k)
t_total = time.perf_counter() - t0
```

Per-query latencies are recorded individually (not just the total) so
percentiles are computed from the actual distribution. A timer that captures
only total time cannot produce a p95.

For **pgvector**, latency is measured **server-side** via the `EXPLAIN
(ANALYZE, FORMAT JSON)` output of each query, taking the `Execution Time`
field. Client-side timing would include the connection round-trip and the
Python driver, which are real costs at the API layer (and are measured in
M4) but are not part of a comparison of ANN libraries. Server-side timing
is what M3's `/readyz` and M4's dashboards will consume; the benchmark
reports the same thing.

The asymmetry is documented in the JSON (`latency_source: "client"` for
FAISS/numpy, `latency_source: "server"` for pgvector) and in the generated
Markdown.

### Seed and randomness

FAISS's HNSW builder is randomized. The benchmark sets:

```python
faiss.omp_set_num_threads(1)   # remove thread-count variance
np.random.seed(42)
random.seed(42)
```

and passes `seed=42` where the FAISS API accepts one. The
`omp_set_num_threads(1)` call is deliberate: on multi-core machines, FAISS's
build time and (in rare cases) the resulting graph vary with thread count.
Pinning to one thread makes the build reproducible at the cost of build
time. On the sample catalog (200 items), the cost is negligible; on a
larger catalog, the benchmark may need to be run with more threads and the
result recorded as thread-count-dependent (a future change; the JSON
records `num_threads` so the choice is visible).

### Output format

Two files:

- `docs/faiss-benchmark.json` — the raw measurement, machine-readable,
  checked into git. It is small (a few KB) and stable across runs that
  produce the same numbers.
- `docs/faiss-benchmark.md` — generated from the JSON by
  `scripts/render_benchmark.py`, with a header that includes the commit,
  the environment summary, and the parameter grid. The Markdown file is
  regenerated, never edited by hand.

The README does **not** embed the JSON. It embeds a Markdown table with
four columns per system (`recall@10`, `ann_recall_vs_exact`, `p95`, `qps`)
at a chosen `ef_search` value, and links to `docs/faiss-benchmark.md` for
the full grid. The chosen `ef_search` is recorded in the README next to
the table so a reader knows which row of the grid is being shown.

### Update policy

- The benchmark is re-run manually when a change could plausibly affect the
  numbers: a new embedding model, a new `hnsw_m` / `ef_construction` pair,
  a pgvector or FAISS upgrade, a hardware change.
- The commit that re-runs the benchmark includes both the JSON and the
  regenerated Markdown, and updates the README table if the numbers moved
  materially (more than 5% on any published cell).
- The README table has a "Last benchmark" line that names the commit and
  the date. A reader can tell how fresh the number is without opening
  `git log`.

The benchmark is **not** a CI gate. It is skipped in CI, and the README
says so. Reasons:

- A CI runner's CPU model, cache size, and RAM are shared with other jobs;
  a latency measured there is a property of the runner, not of the code.
- FAISS pulls a ~30 MB wheel and torch-free but nontrivial install; running
  it on every PR would make the CI slower for a measurement that does not
  change per PR.
- The number the README publishes is one that a reader can reproduce on
  their own machine. A CI-published number invites comparison to a
  different hardware than the reader has.

## Alternatives considered

| Option | Why not |
|---|---|
| **Run FAISS benchmark in CI and gate on it** | Latency numbers from a shared CI runner are not comparable to numbers a reader measures locally, and would drift with the runner's load. The gate would fire on the noise. |
| **Embed benchmark numbers in README by hand without the JSON** | The Markdown is what a reader sees, but the JSON is what a reviewer checks. Without the JSON, the numbers have no provenance. |
| **Generate the README table from the JSON automatically** | Tempting, but the README's table is edited for narrative (which `ef_search` is shown, which columns to include). Auto-generation would either remove that editorial control or require a template system for a table with three rows. The Markdown summary is auto-generated; the README table is written by hand and copied from it. |
| **Measure pgvector latency client-side (like FAISS)** | Client-side time includes the socket round-trip and the Python driver, which are not part of the ANN library's performance. The M4 serving benchmark will measure client-side time, because that is what a user experiences; the FAISS benchmark measures server-side time, because that is what compares libraries. Different questions, different measurements. |
| **Measure FAISS latency server-side (e.g., through a wrapping service)** | Would require standing up an HTTP service around FAISS, which adds the exact latency the benchmark is trying to avoid measuring. Also would compare pgvector's process boundary to a FAISS process boundary with different characteristics. The asymmetry (server-side for pgvector, in-process for FAISS) is deliberate and documented. |
| **Fixed `ef_search` grid rather than a sweep** | A single number hides the recall/latency curve, which is the interesting comparison. A sweep is one extra loop and produces the curve that lets a reader pick an operating point. |
| **`omp_set_num_threads` set to the machine's core count** | Reproducibility requires fixing thread count; the choice of value matters less than the fact that it is fixed. One thread is chosen because the sample catalog is small enough that more threads only add noise. |
| **Record hardware in a separate file (e.g., `HARDWARE.md`)** | The hardware is per-run, not per-project. A run on a different machine produces different numbers and should be recorded alongside those numbers. Putting it in a project-wide file would require the file to change every run. |
| **Skip the exact kNN baseline** | It is the ceiling for recall and the reference for ANN fidelity. Without it, "recall@10 = 0.9" is not interpretable. Also, exact kNN is the fastest thing to run and the most stable reference. |
| **Do not commit the JSON; only commit the Markdown** | The Markdown is generated. Committing only generated output without its input breaks the chain of custody: a reviewer cannot recompute the table from the file in the repo. |

## Consequences

**Positive**

- **Numbers are reproducible.** A reader with the same commit, the same
  vectors, the same golden set, and a comparable machine will get numbers
  close to those in the JSON. The environment block makes "comparable
  machine" checkable.
- **The comparison is honest.** The asymmetry between client-side and
  server-side latency is documented rather than hidden. The recall/latency
  grid shows the curve, not a single point that could have been chosen
  after the fact.
- **Staleness is visible.** The commit hash and date in the JSON and in
  the Markdown header make it easy to tell whether the numbers predate a
  change that would affect them.
- **The README table can be defended.** Every cell traces to a specific
  `ef_search`, a specific commit, and a specific vector set.

**Negative / accepted trade-offs**

- **Manual re-run and manual README update.** A change that affects
  performance does not automatically update the table. Accepted: an
  automatic update would either fire on noise or require a stable benchmark
  environment, which is more infrastructure than the project has.
- **`index_memory_bytes` is best-effort.** RSS is affected by Python
  allocator behavior, page cache, and the state of the process before the
  benchmark starts. The number is directionally correct but not exact.
  Reported as `index_memory_bytes_estimate` in the JSON to avoid implying
  more precision than is available.
- **One-thread FAISS build is slower than it could be.** On a machine with
  many cores, the benchmark takes longer than necessary. For the sample
  catalog this is seconds; for a larger catalog, a future change may relax
  the thread pinning and record the trade-off. The current choice favors
  reproducibility.
- **The chosen `ef_search` in the README is a judgment call.** The grid
  shows four values; the README picks one. The choice is recorded and can
  be changed, but a reader who wants a different point must open the
  Markdown. This is a deliberate editorial decision; the JSON supports any
  choice.
- **The benchmark does not cover filtered queries.** The M2 evaluation does
  (via `docs/adr/0008-filter-strategy.md`), but the FAISS benchmark does
  not, because FAISS's filtered search is a different API
  (`search_with_filtering` or a two-stage approach) and comparing it to
  pgvector's SQL-side filtering is not a like-for-like measurement.
  Filtered-query performance for FAISS is future work if the project
  pursues FAISS beyond the benchmark.

## References

- `docs/adr/0001-pgvector-as-default.md` — why FAISS is a benchmark
  baseline, not a production path
- `docs/adr/0009-golden-set-and-metrics.md` — recall and ANN fidelity
  definitions used by the benchmark
- `docs/adr/0010-evaluation-thresholds.md` — the gate that this benchmark
  does *not* participate in
- `docs/faiss-benchmark.json` — the raw measurement
- `docs/faiss-benchmark.md` — the generated summary
- `scripts/bench_faiss.py` — the benchmark runner
- `scripts/render_benchmark.py` — the Markdown generator
- `README.md` § Performance — where the summary table lives
