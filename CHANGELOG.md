# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `src/recsys/evaluation/`: the offline evaluation harness. `metrics.py`
  (Recall@k, NDCG@k, MRR, ANN fidelity as pure functions), `golden_set.py`
  (loader and validator for `evaluation/golden_set/v1.jsonl`),
  `baselines.py` (random and synthetic popularity; exact kNN is
  `NumpyBackend`), `thresholds.py` (loader and gate), `runner.py`
  (orchestration, query encoding, report assembly). (M2.4)
- `scripts/eval.py`: the CLI that ties the evaluation pieces together.
  Exit codes 0 (gate passed), 1 (gate failed), 2 (could not run), with a
  metrics table printed to stderr so the numbers appear in the CI log.
  (M2.5)
- `evaluation/thresholds.yaml` and `evaluation/thresholds_history.yaml`:
  the initial gate configuration (ADR-0010) and its audit trail. The
  history's second entry records the first measured baseline. (M2.5)
- `tests/unit/test_logging.py`: subprocess-based tests pinning that
  structlog output goes to stderr, never stdout. (M2.5)
- `evaluation` CI job: provisions pgvector, applies migrations, embeds the
  sample catalog, builds and promotes an index, and runs `make eval`. The
  report is uploaded as a build artifact on every run. (M2.5)
- `Makefile` target `eval`. (M2.5)
- `NumpyBackend.embedding_for(item_id)` for the evaluation runner's
  seed-vector lookup. (M2.5)
- `evaluate_gate(..., required_systems=[...])`: a system named in the
  argument that is missing from the report fails the gate. (M2.5)
- `docs/adr/0005-m2-scope-retrieval-only.md` through
  `docs/adr/0012-backend-abstraction.md`: eight ADRs covering M2 scope,
  the pgvector schema and model-swap procedure, index identity and
  collisions, filter strategy and the pgvector version fallback, golden
  set versioning and metrics, evaluation thresholds, FAISS benchmark
  methodology, and backend abstraction. (M2.0)
- `docs/retrieval-and-evaluation.md`: the M2 design doc consolidating
  the eight ADRs. (M2.0)
- `docs/contracts.md` § 1.4 and § 4.4: index registry schema,
  `index_version` format, and retrieval log event names. (M2.0)
- `docs/ops.md`: backup, restore, index lifecycle, disk-space planning,
  and M2-specific failure modes. (M2.0)
- `src/recsys/retrieval/`: `NumpyBackend` (exact kNN, ADR-0012),
  `PgvectorBackend` (HNSW, ADR-0006, ADR-0008), `identity.py` (the
  `index_version` hash, ADR-0007), `filters.py` (the FILTER_FIELDS
  allowlist, ADR-0008), `rerank.py` (an empty `Reranker` protocol so M3
  has a defined plug-in point, ADR-0005), `build.py`, `promote.py`, and
  the class registry that replaced the M0 factory registry. (M2.1–M2.3)
- `migrations/versions/0001_pgvector_schema.py` and the Alembic setup:
  the `item`, `embedding`, and `index_registry` tables with the HNSW
  cosine index. (M2.2)
- `scripts/build_index.py`, `scripts/promote_index.py`,
  `scripts/rollback_index.py`: the index lifecycle CLIs, with a
  `pg_advisory_lock` around promote and rollback. (M2.3)
- `tests/integration/test_index_lifecycle.py` and
  `tests/integration/test_backend_agreement.py`: end-to-end lifecycle
  tests and the backend-agreement regression test from ADR-0012. (M2.3)
- `tests/integration/test_determinism_tiers.py`: the three-tier determinism
  contract is now tested explicitly. Strict (byte-identical Parquet) is
  gated on `RECSYS_STRICT_DETERMINISM=1` and runs in CI; semantic (top-k
  neighbours identical) and tolerance (per-row cosine ≥ 0.9999) run
  everywhere. The CI `test-encoder` job sets the strict env var. (M1.4)
- `docs/embedding-pipeline.md` § 5.1: the tier table now names the test
  that enforces each tier, so the contract and its verification are one
  document. (M1.4)
- `docs/adr/0004-versioned-runs-with-current-pointer.md`: embedding artifacts
  are written into versioned run directories under
  `artifacts/embeddings/runs/<run_id>/`, and the active run is named by a
  one-line text file pointer `artifacts/embeddings/current`. The pointer is a
  file, not a symlink, so the layout is portable across NFS, object storage,
  and container filesystems. The commit protocol is fixed (Parquet → manifest
  → state → pointer), each step is an atomic `os.replace`, and only the final
  step makes a run visible. A `fcntl.flock` on `.lock` serialises runs.
  (M1.3)
- `docs/contracts.md` § 1.3: rewritten for the versioned-run layout, the
  manifest schema, and the `config_hash` definition (parameters that affect
  embedding content, excluding thread count and batch size). (M1.3)
- `docs/embedding-pipeline.md` § 6, § 7: rewritten for the run layout,
  atomic commit protocol, `flock` locking, pre-commit validation, and
  streaming reuse via `ParquetFile.iter_batches` + `ParquetWriter`. (M1.3)
- `docs/adr/0003-plain-python-cli-for-embedding-pipeline.md`: the M1
  pipeline ships as a plain Python CLI; orchestration is deferred to M6 and
  will wrap the CLI, not replace it. (M1.0)
- `docs/embedding-pipeline.md`: M1 design doc — input/output contracts,
  `model_version` and `catalog_snapshot` formats, preprocessing rules, storage
  layout, batch/incremental modes, the three-tier determinism contract
  (strict byte-identical in CI, top-k identical everywhere, cosine tolerance
  locally), sample catalog and golden set, ONNX export lifecycle, and CLI
  contract. (M1.0)
- `docs/contracts.md` § 1.3: embedding artifact formats (`model_version`,
  `catalog_snapshot`, filenames) and the three-tier determinism contract.
  (M1.0)
- `docs/decisions.md`: pre-flight design decisions locked before feature work —
  problem statement, payload vs pipeline split, deterministic/non-deterministic
  idempotency keys, storage tiers, ingestion modes, failure modes, and the
  explicit out-of-scope list. (M0.2)
- `docs/contracts.md`: data, API, config, and telemetry contracts — event
  envelope, recommend request/response, error envelope, config enum table,
  metric/log/span names, and the idempotency matrix per endpoint. (M0.4)
- `src/recsys/` scaffold: config (enum-driven Pydantic Settings with a prod
  guard), API factory with request-context and access-log middleware,
  `/healthz`, `/readyz`, `/metrics`, and 503 stubs for
  recommend/events/churn; structlog JSON logging; Prometheus registry;
  retrieval registry pattern over an `IndexBackend` protocol. (M0.5)
- Test scaffolding under `tests/` with automatic tier markers by directory
  (unit / integration / load). Integration fixtures skip cleanly when
  `RECSYS_TEST_DATABASE_URL` / `RECSYS_TEST_REDIS_URL` are absent. (M0.6)
- `Makefile` with `fmt`, `lint`, `typecheck`, `check-markers`, `test`,
  `test-unit`, `test-integration`, `coverage`, and a `check` gate. (M0.6)
- `.pre-commit-config.yaml` mirroring `make check` plus generic housekeeping
  hooks. (M0.6)
- `scripts/check_markers.py` to enforce test-tier discipline. (M0.6)
- `.github/workflows/ci.yml` with four jobs: `lint`, `test-unit`
  (matrix 3.11/3.12), `test-integration` (pgvector/pgvector:pg16 + redis:7
  service containers), and `docker-build` (image build + smoke test of
  `/healthz`, `/readyz`, `/metrics`). (M0.7)
- Multi-stage `Dockerfile` (builder installs into a virtualenv; runtime runs
  as uid 1000 `recsys`) and `.dockerignore`. (M0.7)
- `docs/adr/0001-pgvector-as-default.md`: why pgvector is the default vector
  store, with FAISS as a benchmark baseline and dedicated vector stores as an
  explicit non-goal. (M0.8)
- `docs/adr/0002-onnx-runtime-for-inference.md`: why `sentence-transformers`
  is used for offline batch embedding but ONNX Runtime serves the online
  encoding path. (M0.8)
- `docs/runbook.md` skeleton: service overview, first-response checklist, and
  four runbooks (readiness failure, p95 latency, fallback-rate spike, SRM in
  a running experiment) plus rollback and escalation. (M0.8)
- `docs/adr/README.md` index and `docs/adr/template.md`. (M0.8)
- `pyproject.toml` as the single source of truth for the package. Runtime
  dependencies live in self-contained optional groups (`api`, `db`, `cache`,
  `auth`, `embeddings`, `observability`, `experiments`, `churn`, `pipeline`,
  `bench`, `load`) with meta groups `service` and `dev`. Also configures Ruff,
  mypy (strict), pytest markers, and coverage. (M0.3)
- `.gitignore` covering secrets, Python build artifacts, test caches,
  notebook checkpoints, DVC, Terraform, Docker overrides, editor noise, and
  Colab scratch directories. Sample data under `data/sample/` is allowlisted.
  (M0.3)
- `.env.example` documenting every configuration variable referenced by the
  README, with backend selectors expressed as enums (`INDEX_BACKEND`,
  `APP_ENV`, `DEVICE`) instead of booleans. (M0.3)
- `CHANGELOG.md` (this file). (M0.3)

### Changed
- `src/recsys/monitoring/logging.py`: structlog now writes to **stderr**,
  and the module-level default (applied at import time) matches what
  `configure_logging` does. This makes CLI scripts safe: their stdout
  carries the final JSON result and nothing else. (M2.5)
- `src/recsys/config/settings.py`: default `EMBEDDING_ONNX_PATH` aligned
  with what `scripts/export_onnx.py` actually writes. (M2.5)
- `README.md`: evaluation table filled with the first measured values from
  the CI `evaluation gate` job; M2 milestone moved to Done. (M2.5)
- `README.md`: relaxed `requires-python` from `>=3.11,<3.12` to `>=3.11`

### Added
- `scripts/bench_faiss.py` and `scripts/render_benchmark.py`: the FAISS
  HNSW benchmark from ADR-0011. Compares exact kNN (numpy) and FAISS HNSW
  over the same vectors, golden set, and query vectors; writes a JSON
  with the environment metadata needed to reproduce the numbers; the
  Markdown is generated from the JSON, never hand-edited. Seed exclusion
  is applied so the benchmark's metrics are comparable to the evaluation
  report's. (M2.6)
- `docs/faiss-benchmark.json` and `docs/faiss-benchmark.md`: the first
  measured benchmark run, on the sample catalog (200 items, dim 384).
  Headline: on 200 items, HNSW is fully connected at every ef_search in
  the grid; ANN fidelity is 1.0000 across the board. On a larger catalog
  the recall/latency trade-off would be visible. (M2.6)

### Fixed
- `src/recsys/evaluation/runner.py`: the evaluation runner now excludes
  seed items from retrieved results before computing metrics. Without
  this, seeds occupied ranks 1..N of every result list (they are the
  nearest neighbours of their own mean) and MRR collapsed to
  `1/(n_seeds + 1)` regardless of model quality. (M2.4)
- `src/recsys/monitoring/logging.py`: structlog output moved from stdout
  to stderr, and the module-level default is now safe for CLI scripts
  that never call `configure_logging`. Before the fix,
  `scripts/build_index.py` polluted its stdout with log lines and broke
  any consumer that parsed it as JSON. (M2.5)
- `scripts/eval.py`: `_env_commit` no longer returns an empty string when
  `GITHUB_SHA` is present but empty, which silently disabled the gate's
  commit check. (M2.5)

### Security
- n/a

## Notes

Milestone tracking lives in [`README.md`](README.md#milestones). Only measured
numbers belong in the README; this changelog records what changed and when.
