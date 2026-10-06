# Embedding pipeline

Design doc for M1. Decisions D2–D10 are recorded here; D1 (pipeline
framework) is an ADR: [`docs/adr/0003-plain-python-cli-for-embedding-pipeline.md`](adr/0003-plain-python-cli-for-embedding-pipeline.md).

> **Status:** locked for M1. Changes that affect artifact formats or the
> determinism contract require an ADR, because other components (M2
> retrieval, M5 evaluation) read these artifacts.

---

## 1. Scope

**In scope for M1**

- Batch and incremental embedding of a catalog of items.
- Versioned artifacts on disk: `model_version` and `catalog_snapshot` in the
  filename and in the artifact manifest.
- Preprocessing rules that are pinned and versioned.
- A three-tier determinism contract, verified by tests.
- A sample catalog and a golden set for offline evaluation.
- A CLI that CI and humans can invoke identically.

**Out of scope for M1**

- Writing embeddings or indexes to PostgreSQL (M2).
- ANN index construction (M2).
- Serving embeddings over HTTP (M3).
- Scheduling and orchestration (M6; see ADR-0003).
- Training or fine-tuning any model.

---

## 2. Inputs and outputs

**Input — catalog**

One JSONL file, one item per line. `data/sample/catalog.jsonl` in the repo is
the canonical sample; larger catalogs are generated or fetched via DVC.

```json
{"item_id": "i_0001", "title": "Dune", "description": "A desert planet epic about spice, prophecy, and empire.", "category": "science_fiction", "brand": "ace_books"}
```

Required fields: `item_id`, `title`, `description`, `category`, `brand`.
Unknown fields are ignored. Missing required fields are a hard error.

**Output — embedding artifacts**

For each `(model_version, catalog_snapshot)` pair:

```
artifacts/embeddings/<model_version>/<catalog_snapshot>.parquet
artifacts/embeddings/<model_version>/<catalog_snapshot>.manifest.json
artifacts/embeddings/<model_version>/<catalog_snapshot>.checksums.json
```

- `*.parquet` — one row per item: `item_id`, `embedding` (list<float32>,
  fixed length), `content_hash`, `preprocessing_version`. No other columns.
- `*.manifest.json` — records `model_version`, `catalog_snapshot`,
  `row_count`, `embedding_dim`, `preprocessing_version`, `created_at`,
  `onnx_artifact_sha256`, `python_version`, `onnxruntime_version`,
  `numpy_version`.
- `*.checksums.json` — `{"parquet_sha256": "<64 hex>"}`.

The manifest exists so that a reviewer (or a future run) can reconstruct
exactly what environment produced this file. Without it, "byte-identical"
becomes unfalsifiable two months later.

---

## 3. Version strings

Both are contracts and are also recorded in
[`docs/contracts.md`](contracts.md) § 1.3.

**`model_version`** — `<label>+<sha8>`

```
minilm-onnx-v1+a3f9e021
```

`<label>` is operator-chosen and human-readable. `<sha8>` is the first 8 hex
characters of the SHA256 of the ONNX artifact bytes. Two artifacts with the
same `<sha8>` are byte-identical. Two artifacts with different `<sha8>` are
different models regardless of `<label>`. The full 64-character hash is
recorded in the manifest; only the 8-character prefix appears in the version
string.

**`catalog_snapshot`** — `sha256:<16 hex>`

```
sha256:9f1e2c8a3b5d7e4f
```

Computed as `sha256("\n".join(f"{item_id}:{content_hash}" for sorted items))`,
truncated to 16 hex characters. Properties that make it useful as an
identifier:

- Row order in the source file does not affect the snapshot.
- A content change on any item (which changes its `content_hash`) changes the
  snapshot.
- Adding or removing items changes the snapshot.
- Two catalog exports with the same snapshot are semantically identical for
  embedding purposes.

**`content_hash` per item** — SHA256 of the *preprocessed* text (see § 4),
truncated to 16 hex characters. Stored as a column in the Parquet so
incremental runs can decide "changed / unchanged" without re-reading the raw
catalog.

---

## 4. Preprocessing contract

One function, `preprocess_item(item: CatalogItem) -> str`. Rules are pinned
and versioned by `preprocessing_version`; any change bumps the version and
therefore the `catalog_snapshot`.

1. Unicode NFKC normalization.
2. Whitespace collapse: `re.sub(r"\s+", " ", text)`.
3. Concatenation: `title + " | " + description`. If `description` is empty,
   only `title` is used (no trailing separator).
4. Truncate to 512 characters (safe upper bound for 512-token MiniLM).
5. `.strip()`.

`PREPROCESSING_VERSION = "v1"` lives in `src/recsys/embeddings/preprocess.py`
and is written into the manifest. Changing any of the five rules above
requires bumping the version; the CI determinism test will then fail on the
new version until it is regenerated deliberately.

---

## 5. Determinism contract

M1's exit criteria is "re-running the pipeline produces identical results".
That phrase is under-specified. This section makes it specific.

### 5.1 Three tiers

| Tier | Assertion | Where |
|---|---|---|
| **Strict** | SHA256 of `*.parquet` matches between two consecutive runs in the same environment | **CI only** |
| **Semantic** | For a fixed probe set of items, the top-k (k=10) nearest neighbours by cosine distance are identical between two runs — same item IDs, same order, ties broken by `item_id` ascending | **CI and local** |
| **Tolerance** | Per-row cosine similarity ≥ 0.9999 between two runs | **Local only** |

### 5.2 Why byte-identical is CI-only

ONNX Runtime and the underlying BLAS (OpenBLAS, MKL, oneDNN, depending on
platform) are permitted to produce numerically different results when thread
counts or kernel selection differ. In practice the difference is on the order
of 1e-7 per element — real for byte comparison, invisible for cosine
similarity. This is a property of the runtime, not of our pipeline.

In CI we remove the variance by pinning:

- `intra_op_num_threads=1` and `inter_op_num_threads=1` on the ONNX session
  options.
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` in the
  environment.
- The `onnxruntime` version (pinned in `pyproject.toml`).
- The model artifact (identified by `sha256` recorded in `model_version`).

With these pins, byte-identical output is **empirically true** in the CI
container, and the strict test asserts it. The claim is honestly scoped:
"byte-identical in the pinned CI environment, verified by test". It is not a
promise that any two machines will produce identical bytes.

In Colab and on developer machines, thread counts vary and the strict
assertion is not made. The semantic and tolerance assertions still run, and
they are what actually protects users: if top-k retrieval changes between two
runs, the system is non-deterministic from the user's perspective regardless
of what the bytes say.

### 5.3 Why top-k matters more than bytes

The user-visible contract is retrieval, not embedding. A pipeline that
produces bit-identical vectors but different top-10 neighbours (e.g. because
of a tie-breaking bug) is broken even though it passes the strict test. The
semantic assertion catches this class of bug. It runs everywhere, on every
test invocation, and is the assertion that gates a merge.

---

## 6. Pipeline modes

Two modes, one primitive.

**Incremental** — default.

- Trigger: item inserted or updated.
- Scope: items whose `content_hash` is absent from the current artifact, or
  differs.
- Output: a new `<catalog_snapshot>` that reflects the updated catalog; the
  previous snapshot's files are left in place (they are the artifact history).

**Batch** — invoked explicitly.

- Trigger: model upgrade, catalog re-import, or a scheduled rebuild.
- Scope: every item in the input catalog.
- Output: a new `<model_version>/<catalog_snapshot>` pair; nothing is
  overwritten.

Both modes call the same `embed_batch(items, encoder) -> np.ndarray`
primitive. Mode decides *which items go in*, not *how they are embedded*. That
keeps the embedding path single-sourced.

CLI:

```
python -m recsys.embeddings.pipeline --mode=incremental \
  --catalog data/sample/catalog.jsonl \
  --out artifacts/embeddings
```

Exit codes: `0` success, `2` input error (missing/invalid catalog), `3`
encoder error (ONNX load failure), `4` output error (disk full, permission).
The stdout summary is a single JSON line:

```json
{"event": "pipeline.done", "mode": "incremental", "model_version": "minilm-onnx-v1+a3f9e021", "catalog_snapshot": "sha256:9f1e2c8a3b5d7e4f", "row_count": 200, "embedding_dim": 384, "duration_ms": 4123, "parquet_sha256": "..."}
```

The JSON line is the integration surface for CI and for M6 orchestration.

---

## 7. Storage layout

```
artifacts/
├── onnx/
│   └── <model_slug>/
│       ├── model.onnx            # the exported graph (mean-pooling + L2-normalize folded in)
│       ├── model.onnx.sha256     # SHA256 of model.onnx; loaded and re-verified at runtime
│       ├── tokenizer.json        # fast-tokenizer file; runtime tokenizes without transformers
│       └── config.json           # {model_name, max_seq_length, embedding_dim}
└── embeddings/
    └── <model_version>/
        ├── <catalog_snapshot>.parquet
        ├── <catalog_snapshot>.manifest.json
        └── <catalog_snapshot>.checksums.json
```

`model_slug` is the model name with `/` replaced by `__` and any other
filesystem-hostile character removed. Example:
`sentence-transformers/all-MiniLM-L6-v2` → `sentence-transformers__all-MiniLM-L6-v2`.

`artifacts/` is git-ignored. Sample artifacts (small, deterministic) can be
committed under `data/sample/artifacts/` for tests; the real catalog's
artifacts live in DVC or an object store (see `docs/decisions.md` § 4).

---

## 8. Sample catalog and golden set

**Sample catalog** — `data/sample/catalog.jsonl`, committed.

- 20 topics × 10 items = **200 items**.
- Topics span varied domains (sci-fi books, jazz albums, running shoes,
  mechanical keyboards, cooking, travel guides, etc.).
- Each item: 30–60 words of title + description, plus `category` and `brand`.
- Deterministic generation: the file is authored, not generated at test time.
  Reviewers can read it in a browser; diffs are meaningful.

**Golden set** — `evaluation/golden_set/queries.yaml`, committed.

- 20 queries, one per topic.
- Each query has 3–5 seed items from its topic.
- Every *other* item in the same topic is labelled relevant.
- Ground truth is objective: the labels test whether the model clusters by
  topic, which a pre-trained sentence encoder should do well on clean
  synthetic data. If Recall@10 for a topic is below the threshold chosen in
  M2, the model or the preprocessing is wrong, and the failure is
  falsifiable.

Larger golden sets for M2 are versioned via DVC, not committed.

---

## 9. ONNX export lifecycle

Export is a separate target, not part of `make embed`:

```
make export-onnx MODEL=sentence-transformers/all-MiniLM-L6-v2
```

- Loads the sentence-transformers model, exports to ONNX, verifies output
  parity against the reference on a fixed input set within tolerance, and
  writes `artifacts/onnx/<model_slug>/model.onnx` plus its SHA256 sidecar.
- Skips if `model.onnx` exists and its SHA256 matches. Re-export requires
  deleting the artifact or passing `--force`.
- The exported artifact's SHA256 determines `<sha8>` in `model_version`.

The parity test (ONNX vs sentence-transformers) runs in the integration tier
and is required by ADR-0002.

---

## 10. What lands in M2

- `make index-build` — reads the Parquet artifacts, inserts rows into
  pgvector with an HNSW index, records `index_version`.
- Golden-set evaluation: Recall@k, NDCG@k, MRR — the sample golden set is the
  fast fixture used in CI; the full set runs before index promotion.
- FAISS benchmark against the same artifacts.
- The offline evaluation gate (`make eval`) with thresholds in
  `evaluation/thresholds.yaml`.

Nothing in M2 changes the artifact format defined here. That stability is the
point of writing this document before the code.
