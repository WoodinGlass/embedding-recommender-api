# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- n/a

### Fixed
- n/a

### Security
- n/a

## Notes

Milestone tracking lives in [`README.md`](README.md#milestones). Only measured
numbers belong in the README; this changelog records what changed and when.
