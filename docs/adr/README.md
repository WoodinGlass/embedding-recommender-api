# Architecture Decision Records

An ADR captures a decision that is **significant enough to be hard to reverse**
and **non-obvious enough to need justification**. Code comments explain *what*;
ADRs explain *why, versus what else*.

## Convention

- Filename: `NNNN-kebab-case-title.md`, starting at `0001`.
- Status is one of: `Proposed`, `Accepted`, `Rejected`, `Deprecated`,
  `Superseded by ADR-NNNN`.
- Once accepted, an ADR is **immutable**. Changed circumstances produce a new
  ADR that supersedes the old one; the old one stays for the historical record.
- Copy `template.md` to start a new ADR.

## When an ADR is required

Per `docs/contracts.md` § 6 (Change management):

- **Additive change** (new field, new metric, new enum value in a
  non-exhaustive position) — no ADR.
- **Behavioral change** (same shape, different semantics) — ADR required.
- **Breaking change** (removed field, changed type, removed enum value) —
  ADR required, plus an API path version bump.
- **Anything locked in `docs/decisions.md`** — new ADR required to change it.
- **New external dependency or infrastructure component** — ADR required.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-pgvector-as-default.md) | pgvector as the default vector store | Accepted |
| [0002](0002-onnx-runtime-for-inference.md) | ONNX Runtime for embedding inference | Accepted |
| [0003](0003-plain-python-cli-for-embedding-pipeline.md) | Plain Python CLI for the embedding pipeline | Accepted |
| [0004](0004-versioned-runs-with-current-pointer.md) | Versioned run directories with an atomic `current` pointer | Accepted |
| [0005](0005-m2-scope-retrieval-only.md) | M2 scope is retrieval only; re-ranking moves to M3 | Accepted |
| [0006](0006-pgvector-schema.md) | pgvector schema, fixed embedding dimension, and the model-swap procedure | Accepted |
| [0007](0007-index-version-identity.md) | `index_version` identity and collision handling | Accepted |
| [0008](0008-filter-strategy.md) | Filtered ANN search — iterative scan, fallback, and version detection | Accepted |
| [0009](0009-golden-set-and-metrics.md) | Golden set versioning, metrics, query encoding, and baselines | Accepted |
| [0010](0010-evaluation-thresholds.md) | Evaluation thresholds — absolute floor, history, and the CI gate | Accepted |
| [0011](0011-faiss-benchmark-methodology.md) | FAISS benchmark methodology and reproducibility | Accepted |
| [0012](0012-backend-abstraction.md) | Backend abstraction, `NumpyBackend` for exact kNN, and pgvector in CI only | Accepted |
