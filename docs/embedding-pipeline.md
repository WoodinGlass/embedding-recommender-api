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
## 2. Inputs and outputs

**Input — catalog**

One JSONL file, one item per line. `data/sample/catalog.jsonl` in the repo is
the canonical sample; larger catalogs are generated or fetched via DVC.

```
{"item_id": "i_0001", "title": "Dune", "description": "A desert planet epic about spice, prophecy, and empire.", "category": "science_fiction", "brand": "ace_books"}
```

Required fields: `item_id`, `title`, `description`, `category`, `brand`.
Unknown fields are ignored. Missing required fields are a hard error.

**Output — embedding artifacts**

Each pipeline run writes a **versioned run directory** under
`artifacts/embeddings/runs/`, plus a small file pointer
`artifacts/embeddings/current` that names the active run. The full layout,
the manifest schema, and the `config_hash` definition are specified in
`docs/contracts.md` § 1.3; the atomicity guarantees are the subject of
[ADR-0004](adr/0004-versioned-runs-with-current-pointer.md).

```
artifacts/embeddings/
├── runs/
│   └── <ISO8601-Z>__<model_version>/
│       ├── embeddings.parquet      # one row per item (the contract)
│       ├── manifest.json           # metadata about the run
│       └── state.json              # incremental sidecar (§ 7.1)
├── current                         # one line: the active run_id
└── .lock                           # fcntl.flock target
```

- `embeddings.parquet` — one row per item: `item_id`, `embedding`
  (list<float32>, fixed length), `content_hash`, `preprocessing_version`.
  No other columns. Rows are sorted by `item_id`.
- `manifest.json` — the source of truth for what the run contains:
  `model_version`, `preprocessing_version`, `config_hash`,
  `catalog_snapshot`, `onnx_artifact_sha256`, `parquet` (path, sha256,
  rows, encoded_rows, dim, dtype), `state` (path, sha256), `environment`
  (python, onnxruntime, numpy, pyarrow versions), and `created_at`.
  `created_at` makes the manifest *not* byte-stable, by design; the strict
  determinism tier in § 5.1 applies to the Parquet only.
- `state.json` — a supporting file for incremental runs (§ 7.1). Not part
  of the contract.
- `current` — a one-line text file containing the active `run_id`. Written
  last in the commit sequence. This is what makes a run visible.

The manifest exists so a reviewer (or a future run) can reconstruct exactly
what environment produced this file. Without it, "byte-identical" becomes
unfalsifiable two months later.

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

| Tier | Assertion | Where | Enforced by |
|---|---|---|---|
| **Strict** | SHA256 of `*.parquet` matches between two consecutive runs in the same environment | **CI only** | `tests/integration/test_determinism_tiers.py::test_strict_tier_two_batch_runs_produce_identical_bytes` (gated on `RECSYS_STRICT_DETERMINISM=1`) |
| **Semantic** | For a fixed probe set of items, the top-k (k=10) nearest neighbours by cosine distance are identical between two runs — same item IDs, same order, ties broken by `item_id` ascending | **CI and local** | `tests/integration/test_determinism_tiers.py::test_semantic_tier_topk_identical` |
| **Tolerance** | Per-row cosine similarity ≥ 0.9999 between two runs | **CI and local** | `tests/integration/test_determinism_tiers.py::test_tolerance_tier_per_row_cosine` |

The strict tier is enabled by setting the environment variable
`RECSYS_STRICT_DETERMINISM=1`. CI does this on the `test-encoder` job
(see `.github/workflows/ci.yml`). Locally, the strict test skips with an
explanatory reason; the semantic and tolerance tiers always run.

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
## 6. Pipeline modes

Two modes, one primitive.

**Incremental** — default.

- Trigger: item inserted or updated.
- Scope: items whose `content_hash` differs from the value recorded in the
  current `state.json` (§ 7.1), plus any item not present in that state.
- Output: a new run directory whose Parquet contains every item in the
  catalog — reusing embeddings from the previous Parquet for unchanged items,
  encoding only the changed ones. Previous run directories are left in place
  (they are the artifact history, and they are the rollback target).
- Fallback: if `state.json` is missing, malformed, or its `config_hash`,
  `model_version`, or `preprocessing_version` do not match the current
  invocation, every item is treated as changed. The run is still correct,
  just slower.
- Idempotency: if the catalog snapshot, config hash, and per-item content
  hashes all match the currently active run, the pipeline writes nothing,
  leaves `current` untouched, exits 0, and reports `"status": "no_changes"`.

**Batch** — invoked explicitly.

- Trigger: model upgrade, catalog re-import, or a scheduled rebuild.
- Scope: every item in the input catalog.
- Output: a new run directory. Previous runs are not modified.
- Idempotency: batch always writes a new run, even if nothing changed. This
  preserves the strict determinism contract: two batch runs of the same
  catalog under the same pinned environment must produce byte-identical
  Parquet, which requires a fresh write each time.

Both modes call the same `embed_batch(items, encoder) -> np.ndarray`
primitive. Mode decides *which items go in*, not *how they are embedded*. That
keeps the embedding path single-sourced.

CLI:

```
python -m recsys.embeddings.pipeline --mode=incremental \
  --catalog data/sample/catalog.jsonl \
  --out artifacts/embeddings
```

Exit codes: `0` success (including `no_changes`), `2` input error
(missing/invalid catalog), `3` encoder error (ONNX load failure), `4` output
error (disk full, permission, validation failure), `5` lock timeout
(another pipeline run holds `.lock`).

The stdout summary is a single JSON line:

```
{"event":"pipeline.done","status":"ok","mode":"incremental","run_id":"2026-10-07T11-30-00Z__minilm-onnx-v1+a3f9e021","model_version":"minilm-onnx-v1+a3f9e021","config_hash":"sha256:...","catalog_snapshot":"sha256:9f1e2c8a3b5d7e4f","total":200,"unchanged":200,"changed":0,"new":0,"deleted":0,"encoded":0,"duration_ms":89,"output_path":"artifacts/embeddings/runs/2026-10-07T11-30-00Z__minilm-onnx-v1+a3f9e021/embeddings.parquet","checksum":"..."}
```

The JSON line is the integration surface for CI and for M6 orchestration.
Per-batch progress lines go to **stderr** (also JSON), so stdout stays a
single line and can be piped into `jq` without filtering.

## 7. Storage layout
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
    ├── runs/
    │   └── <run_id>/
    │       ├── embeddings.parquet
    │       ├── manifest.json
    │       └── state.json
    ├── current                   # one-line text file: the active run_id
    └── .lock                     # fcntl.flock target
```

`model_slug` is the model name with `/` replaced by `__` and any other
filesystem-hostile character removed. Example:
`sentence-transformers/all-MiniLM-L6-v2` → `sentence-transformers__all-MiniLM-L6-v2`.

`run_id` is `<ISO8601-Z>__<model_version>`. Both components are also stored
inside `manifest.json`, so parsing the run_id is never required.

`artifacts/` is git-ignored. Sample artifacts (small, deterministic) can be
committed under `data/sample/artifacts/` for tests; the real catalog's
artifacts live in DVC or an object store (see `docs/decisions.md` § 4).

### 7.1 Incremental state file

Incremental runs need to decide, per item, whether its content has changed
since the last run. Reading the last Parquet in full and re-hashing every row
is the wrong trade: for a 1M-item catalog, that reads ~1.5 GB just to answer
"which 100 items changed?". Full memory loading is worse still: a 1M x 384
float32 table is 1.5 GB in the best case and 3–4 GB during conversion.

Instead, each successful run writes a small sidecar inside its run directory:

```
artifacts/embeddings/runs/<run_id>/state.json
```

Shape:

```
{
  "schema_version": 1,
  "model_version": "minilm-onnx-v1+a3f9e021",
  "preprocessing_version": "v1",
  "config_hash": "sha256:...",
  "hash_algorithm": "sha256",
  "catalog_snapshot": "sha256:9f1e2c8a3b5d7e4f",
  "items": {
    "i_0001": "abcdef0123456789",
    "i_0002": "0123456789abcdef"
  }
}
```

The state is considered **valid for reuse** only if all of the following
match the current invocation:

1. `schema_version == 1`
2. `model_version` equals the encoder's `model_version`
3. `preprocessing_version` equals `PREPROCESSING_VERSION`
4. `config_hash` equals the current config hash (§ 1.3 of `docs/contracts.md`)

If any do not match, every item is treated as changed (an "invalidated"
state). This is what makes a model upgrade or a preprocessing change
correctly trigger a full re-encode even when the catalog text is unchanged.

`state.json` is *not* part of the artifact contract — it is a cache that can
be deleted and rebuilt by running `--mode=batch`. The contract for downstream
consumers is `embeddings.parquet` and `manifest.json`. If `state.json` is
missing, malformed, or invalidated, incremental mode falls back to treating
every item as changed (logging a `pipeline.state.invalidated` warning with
the specific mismatch), which is correct but slower.

State is written **atomically**: the pipeline writes to a temp file in the
same directory, `fsync`, then `os.replace`. It is also written **after** the
Parquet and manifest, never before, so a crash cannot leave the state
pointing at an artifact that does not exist (see § 7.2).

### 7.2 Commit protocol (atomicity)

The pipeline commits a run in four ordered steps, each of which is an
atomic write (`tempfile` in the target directory → `fsync` → `os.replace`):

1. `runs/<run_id>/embeddings.parquet`
2. `runs/<run_id>/manifest.json`
3. `runs/<run_id>/state.json`
4. `current` (the file pointer)

Only step 4 makes the run visible. A crash before step 4 leaves the previous
`current` in place; the new run directory is orphaned and will be ignored by
consumers. A crash between steps 2 and 3 leaves the Parquet and manifest in
place but no state; the next incremental run sees the previous state (or
none) and re-encodes whatever it cannot prove is unchanged. This is safe by
construction.

The inverse ordering (write `current` first, or write state before Parquet)
would allow a consumer to read a manifest whose Parquet is missing, or an
incremental run to skip items whose new embeddings were never written. Both
are prevented by the fixed order above.

### 7.3 Locking

Every pipeline invocation takes an exclusive `fcntl.flock` on
`artifacts/embeddings/.lock` before reading state or writing anything. If the
lock is held, the CLI retries briefly (default 30 s, configurable with
`--lock-timeout`) and then exits with code `5` and a
`{"event":"pipeline.locked"}` line on stderr.

`flock` is auto-released by the kernel when the process dies, so there is no
stale-lock problem. **NFS is not supported**: `flock` semantics are not
reliable on NFS and many network filesystems. Moving artifacts to NFS, EFS,
or an object store will require a different locking strategy — a separate
ADR.

### 7.4 Validation before commit

Before writing anything, the pipeline validates the assembled table:

- `embeddings.dtype == np.float32`
- `embeddings.shape == (n_items, encoder.embedding_dim)`
- `np.isfinite(embeddings).all()` (no NaN, no Inf)
- `len(set(item_ids)) == len(item_ids)` (unique IDs)
- `np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)`
- `len(item_ids) == len(content_hashes) == manifest.rows`

A failure raises `OutputError` and no files are written. This catches the
class of bugs — off-by-one in the merge, wrong pooling, model loaded with the
wrong dimension — that would otherwise reach M2 retrieval and produce
plausible-looking but wrong results.

### 7.5 Streaming

Reuse does not load the previous Parquet into memory. The pipeline opens the
previous Parquet with `pyarrow.parquet.ParquetFile.iter_batches`, walks it
batch by batch, and for each batch either forwards the rows (unchanged
items) or drops them (changed or deleted items). New embeddings are produced
in the same batch size and interleaved before writing. Peak memory is
`batch_size * dim * 4 bytes`, i.e. ~77 MB at batch size 50k and dim 384.

Writing uses `pyarrow.parquet.ParquetWriter`, one row group per output
batch, with `zstd` compression. No intermediate fully-materialised array is
created.

## 8. Sample catalog and golden set

**Sample catalog** — `data/sample/catalog.jsonl`, committed.

- 20 topics × 10 items = **200 items**.
- Topics span varied domains (sci-fi books, jazz albums, running shoes,
  mechanical keyboards, cooking, travel guides, etc.).
- Each item: 30–60 words of title + description, plus `category` and `brand`.
- Deterministic generation: the file is authored, not generated at test time.
  Reviewers can read it in a browser; diffs are meaningful.

**Golden set** — `evaluation/golden_set/v1.jsonl`, committed. Format and versioning are fixed by ADR-0009.

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
