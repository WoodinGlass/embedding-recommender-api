# ADR-0006: pgvector schema, fixed embedding dimension, and the model-swap procedure

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

M2 introduces the first persistent schema in the project. Up to this point,
all artifacts live on disk under `artifacts/` and the only database tables
are the ones M0 scaffolded (none, in fact — `/readyz` currently reports ready
unconditionally). M2 adds three tables: `item`, `embedding`, and
`index_registry`. The schema is the first place where an embedding model's
dimensionality is recorded in a system that is otherwise schema-free, and the
first place where a decision about the *shape* of stored vectors has
long-term consequences.

The relevant constraints:

- pgvector's `vector(N)` type fixes the dimension at DDL time. `vector` with
  no argument accepts any dimension but disables the HNSW index (which
  requires a fixed dimension) and removes a database-level guardrail.
- pgvector does **not** support `ALTER TABLE ... ALTER COLUMN ... TYPE
  vector(M)` when changing dimension. The column must be dropped and
  recreated, and the data re-inserted. This is a documented limitation of the
  extension, not a bug in our usage.
- The project's history (M1) records `model_version`, `catalog_snapshot`,
  and `preprocessing_version` for every embedding set. Those identifiers do
  not appear in the database schema proposed for M2 unless we put them there
  deliberately.
- `README.md` § Retrieval mentions a `language` filter field. If the schema
  omits it, the filter cannot be served and the README becomes a lie.
- Incremental embedding runs (M1) produce new content hashes. Without an
  `updated_at` on `item`, the age of a stored row is unknowable and
  observability has nothing to key on.

## Decision

### Schema

Three tables, one extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE item (
    item_id        TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    description    TEXT NOT NULL,
    category       TEXT NOT NULL,
    brand          TEXT NOT NULL,
    language       TEXT NOT NULL DEFAULT 'en',
    content_hash   TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX item_category_idx ON item (category);
CREATE INDEX item_brand_idx    ON item (brand);
CREATE INDEX item_language_idx ON item (language);

CREATE TABLE embedding (
    item_id         TEXT NOT NULL REFERENCES item (item_id) ON DELETE CASCADE,
    index_version   TEXT NOT NULL REFERENCES index_registry (index_version)
                    ON DELETE CASCADE,
    model_version   TEXT NOT NULL,
    vector          vector(384) NOT NULL,
    PRIMARY KEY (item_id, index_version)
);
CREATE INDEX embedding_hnsw_cosine_idx
    ON embedding USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE index_registry (
    index_version          TEXT PRIMARY KEY,
    model_version          TEXT NOT NULL,
    catalog_snapshot       TEXT NOT NULL,
    preprocessing_version  TEXT NOT NULL,
    metric                 TEXT NOT NULL CHECK (metric IN ('cosine', 'l2', 'ip')),
    hnsw_m                 INTEGER NOT NULL,
    hnsw_ef_construction   INTEGER NOT NULL,
    hnsw_ef_search         INTEGER NOT NULL,
    pgvector_version       TEXT NOT NULL,
    golden_set_version     TEXT NOT NULL,
    row_count              INTEGER NOT NULL,
    status                 TEXT NOT NULL CHECK (status IN ('building', 'active', 'retired')),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    activated_at           TIMESTAMPTZ,
    retired_at             TIMESTAMPTZ
);

-- At most one active index at any moment.
CREATE UNIQUE INDEX index_registry_one_active
    ON index_registry (status) WHERE status = 'active';
```

### Fixed dimension

`vector(384)` is fixed to the embedding dimension of the model pinned in
`EMBEDDING_MODEL` (`sentence-transformers/all-MiniLM-L6-v2`, 384 dims). The
alternative — untyped `vector` — is rejected for three reasons:

1. **HNSW requires a fixed dimension.** The whole point of the retrieval
   path in M2 is an HNSW index. An untyped column cannot be indexed with
   HNSW, so the query planner falls back to a sequential scan, which defeats
   the latency target.
2. **Database-level validation.** A `vector(384)` column rejects a wrong-
   shaped insert at commit time, before it reaches the retrieval layer. This
   complements, and does not replace, the application-side
   `validate_embeddings` check from M1.
3. **Index build parameters are dimension-dependent.** `m` and
   `ef_construction` interact with dimensionality. A schema that pretends
   dimensionality is a runtime variable makes it harder to reason about
   build parameters and about the exact recall/latency curve that M2
   publishes.

### Model swap procedure

Changing the embedding model changes the dimension. Because pgvector does
not support `ALTER COLUMN ... TYPE vector(M)`, the procedure is a **new
index version, new column dimension, in-place swap**, executed as a
maintenance operation with a documented ADR:

1. Write a new ADR: "swap embedding model from X (dim D_x) to Y (dim D_y)",
   with the reason, the new `model_version`, and the expected index build
   time.
2. Add a new Alembic migration that:
   - `ALTER TABLE embedding DROP COLUMN vector;`
   - `ALTER TABLE embedding ADD COLUMN vector vector(D_y);`
   - Recreates `embedding_hnsw_cosine_idx`.
   The old rows are intentionally dropped: they are incompatible with the
   new dimension and are reconstructed from the new artifact run.
3. Run `make index-build` with the new model. This writes a new
   `index_version` in `building` status and populates `embedding`.
4. Run `make eval` on the new index. Thresholds are **recomputed** (or at
   least reviewed) because the golden set is unchanged but the model is not.
5. `make index-promote VERSION=<new>`. The old index is marked `retired`;
   its rows remain in `embedding` until a prune target is run.

This procedure is deliberately heavyweight. A model swap is rare and its
effects are large; a schema that made the operation *look* cheap would hide
the cost.

## Alternatives considered

| Option | Why not |
|---|---|
| **Untyped `vector` column** | HNSW cannot index it; sequential scan is the planner fallback. Directly contradicts the M2 latency requirement. Also removes the database-level dimension guardrail. |
| **`vector(384)` but store the dimension in `index_registry` only** | The dimension is fixed at DDL time, so storing it at runtime is documentation, not enforcement. Recorded values can drift from the actual column type. |
| **Separate table per model (`embedding_384`, `embedding_768`)** | Multiplies migrations and index definitions. The pgvector extension is not designed around this pattern; query planning and privileges become per-table concerns. Rejected for complexity. |
| **Store the vector in `BYTEA` and cast to `vector` at query time** | Removes the pgvector index entirely. Loses the HNSW performance that M2 exists to measure. |
| **Use the `vector` type without a length, plus a CHECK constraint on `vector_dims(vector) = 384`** | CHECK on a per-row function call is evaluated on write, but the HNSW index still cannot be created on a variable-length column. Same planner failure as untyped `vector`. |
| **Model swap via in-place `UPDATE` with casting** | pgvector does not support the cast across dimensions (padding/truncation would be required, which is semantically wrong for embeddings). |

## Consequences

**Positive**

- HNSW indexes the column. This is the prerequisite for the M2 latency
  measurement being meaningful.
- Wrong-shaped inserts fail at the database, before they reach the retrieval
  code path. Bugs surface closer to their cause.
- The `index_registry` records every parameter that affects the recall/
  latency curve — including the pgvector extension version and the HNSW
  `ef_search` default — so M2's published numbers can be reproduced from
  the registry alone.
- The `language`, `updated_at`, `content_hash`, `metric`, and
  `golden_set_version` fields close gaps identified in review. Without them
  either a filter referenced by the README cannot be served, or a threshold
  cannot be tied to the golden set that produced it.

**Negative / accepted trade-offs**

- **Model swaps are heavyweight.** Dropping and recreating the vector column
  discards all stored embeddings and requires a full re-embed and index
  build. Accepted: the operation is rare, and hiding the cost behind an
  in-place ALTER would mislead the operator.
- **The extension version is baked into the schema.** Moving to a pgvector
  release that changes on-disk format requires a fresh build, not an
  in-place migration. The `pgvector_version` column in `index_registry`
  makes the current version explicit and comparable across runs.
- **`embedding` rows accumulate.** Blue/green swaps leave the retired
  index's rows in place for instant rollback. At the target scale (1M items
  × 2 indexes = 2M rows × 384 floats ≈ 3 GB) this is acceptable. A prune
  target will be added when it stops being acceptable; the decision and its
  trigger are recorded rather than deferred silently.
- **The `vector(384)` type appears in the Alembic migration.** Changing it
  requires a new migration, not a config change. This is the correct
  trade-off for a portfolio project, where "the schema says exactly what it
  expects" is more valuable than "the schema bends to config".

## References

- `docs/retrieval-and-evaluation.md` — the M2 design doc
- `docs/contracts.md` § 2.1 — the allowlist of filter fields referenced by
  the schema
- `docs/adr/0001-pgvector-as-default.md` — why pgvector, why HNSW
- `docs/adr/0007-index-version-identity.md` — how `index_version` is
  computed from the registry fields
- `migrations/versions/` — the Alembic migration implementing this schema
