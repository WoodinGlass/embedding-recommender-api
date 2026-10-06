# ADR-0002: ONNX Runtime for embedding inference

- **Status:** Accepted
- **Date:** 2026-10-07
- **Deciders:** project maintainer

## Context

Two encoding paths exist in this service:

1. **Offline batch** (M1): embed the whole catalog with a
   `sentence-transformers` model.
2. **Online query encoding** (M3): encode a user's seed items on the request
   path, before ANN retrieval.

The offline path runs once per catalog snapshot and can afford a heavy
dependency. The online path cannot: it shares CPU with retrieval, must stay
within the p95 < 200 ms budget (README § Performance), and its runtime is
what gets shipped in the Docker image.

The current scaffold (`pyproject.toml`) separates the two concerns through
extras: `[embeddings]` pulls `sentence-transformers` and `onnxruntime`;
`[api,observability]` pulls only what the service needs at runtime.

## Decision

- **Offline**: use `sentence-transformers` as the reference implementation.
  The model version pinned in config (`EMBEDDING_MODEL`) is the single source
  of truth for what was embedded.
- **Online**: export the same model to **ONNX** and serve it with
  `onnxruntime`. The service runs with `[api,observability]` and loads the
  ONNX artifact from `EMBEDDING_ONNX_PATH`.
- **Parity is verified in tests**: an integration test compares ONNX output
  to the reference model output on a fixed set of inputs, within a documented
  tolerance. The reference model is a test-only dependency.

## Alternatives considered

| Option | Why not |
|---|---|
| **`sentence-transformers` directly at runtime** | Pulls `torch` into the runtime image (~800 MB), adds ~1–2 s to cold start, and is slower on CPU than ONNX Runtime for the same model. Every container start pays this cost; the online path only needs the forward pass. |
| **`transformers` + `torch` with TorchScript** | TorchScript export is a one-way process that lags behind model updates, and the runtime still carries torch. The ONNX ecosystem is more broadly supported for CPU inference. |
| **CTranslate2** | Fast and lightweight, but its supported model zoo is narrower, and it introduces a third export pipeline alongside the reference. ONNX is the more standard intermediate. |
| **TensorRT** | GPU-only. The deployment target (`docs/decisions.md` § 4) is a single CPU instance; GPU adds cost without changing the p95 target, which the CPU path meets. |
| **Precompute embeddings for seed items at ingest time** | Removes encoding from the hot path entirely, but only works when every user's seed set is known in advance. Cold-start and long-tail seed items still need online encoding. Kept as a cache-layer optimization (M3), not a replacement. |

## Consequences

**Positive**

- Runtime image excludes `torch`, keeping it small and cold start under a
  second. This is what makes `docker compose up` on a laptop viable.
- CPU inference is measurably faster than the reference `torch` path for the
  same model, which directly buys margin against the p95 target.
- The ONNX artifact is immutable per `model_version`, which fits the
  deterministic embedding contract (`docs/decisions.md` § 3): the same
  `model_version` always produces the same embedding.
- Model and runtime are decoupled: upgrading `onnxruntime` does not change
  the model, and re-exporting the model does not require a runtime upgrade.

**Negative / accepted trade-offs**

- The ONNX export is an extra pipeline step (`pipelines/` in M1). It must be
  scripted, versioned, and covered by parity tests. Skipping that means
  serving embeddings that differ from the reference with no way to notice.
- The parity test needs `sentence-transformers` as a test-only dependency, so
  the integration test suite is heavier than a pure-ONNX test suite would be.
  This is accepted: correctness relative to the reference matters more than
  test-suite weight.
- Dynamic input shapes (variable batch size, variable sequence length) require
  explicit handling in the export. The export script must pin the axes it
  intends to allow and the parity test must cover the boundary shapes.
- ONNX Runtime's numerical output is bit-identical to `torch` for the tested
  shapes in practice, but this is a property of the current model and runtime,
  not a guarantee. The parity test enforces it per model version.

## References

- `docs/decisions.md` § 3 (determinism), § 4 (storage tiers)
- `docs/contracts.md` § 3 (`EMBEDDING_MODEL`, `EMBEDDING_ONNX_PATH`, `DEVICE`)
- `pyproject.toml` extras: `[embeddings]`, `[api]`
- Parity test and export script: added in M1
