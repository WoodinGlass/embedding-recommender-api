# ADR-0007: `index_version` identity and collision handling

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

The retrieval layer needs a stable, human-readable identifier for a built
index. This identifier appears in three places:

1. As a foreign key in `embedding.index_version`, so multiple indexes can
   coexist in the same table (blue/green).
2. In `RecommendResponse.meta.index_version`, so a client can correlate a
   response with the exact index that produced it (`docs/contracts.md` § 2.1).
3. In logs and metrics (`recsys_active_index_info{index_version=...}`), so
   the observable behavior of one index can be distinguished from another.

The identifier must satisfy four properties:

- **Deterministic.** Two builds with the same inputs produce the same id.
  Without this, "rebuild the same index" produces a new id and blows up the
  registry.
- **Content-addressed.** Two builds with different inputs produce different
  ids, including inputs that do not appear in the artifact's bytes — such
  as `metric` and `pgvector_version`.
- **Short enough for logs.** `recsys_active_index_info` is a Prometheus
  label. Labels appear in every scrape; a 64-hex-character label is wasteful
  and easy to misread.
- **Collision-safe.** An id truncated to a few characters has a non-zero
  probability of colliding. Truncation without collision handling is a latent
  bug that appears only at scale.

The pattern established in M1 for `model_version` and `catalog_snapshot` is
`<human-label>+<sha-prefix>` or `sha256:<hex>`. This ADR applies the same
principle to the index.

## Decision

### Identifier format

```
idx-<sha-prefix>
```

where `<sha-prefix>` is the first 8 hex characters of the SHA256 of a
canonical JSON object containing every input that affects the retrieval
result. Example:

```
idx-a3f9e021
```

Total length: 12 characters. Short enough for a Prometheus label, long
enough to be unambiguous in practice.

### Hash inputs

The hash covers every parameter that changes which items a query returns,
in which order:

```
{
  "model_version": "minilm-onnx-v1+a3f9e021",
  "catalog_snapshot": "sha256:9f1e2c8a3b5d7e4f",
  "preprocessing_version": "v1",
  "metric": "cosine",
  "hnsw_m": 16,
  "hnsw_ef_construction": 64,
  "pgvector_version": "0.8.0"
}
```

Each component is included for a specific reason:

| Component | Why it affects results |
|---|---|
| `model_version` | Different models produce different vectors; the vectors determine which items are nearest. |
| `catalog_snapshot` | A different catalog is a different set of points; the nearest neighbours of a fixed query change if the set changes. |
| `preprocessing_version` | Preprocessing changes the text that goes into the model, which changes the vectors. Same model + different preprocessing = different vectors. |
| `metric` | Cosine, L2, and inner-product distance orderings disagree in general. Same vectors + different metric = different neighbours. |
| `hnsw_m` | The number of connections per node changes recall at a given `ef_search`. |
| `hnsw_ef_construction` | The size of the candidate set during build changes the quality of the graph, hence recall. |
| `pgvector_version` | HNSW implementation details are not frozen across pgvector releases. A graph built by 0.7 is not bit-identical to one built by 0.8, even with the same inputs. |

Deliberately **excluded**:

- `hnsw_ef_search` — it is a query-time knob. Changing it does not require
  a rebuild; it changes the operating point on the recall/latency curve of
  the same index. It is recorded in `index_registry` as the default used at
  evaluation time, but is not part of the identity.
- `golden_set_version` — evaluation metadata, not a build input. Recorded
  in `index_registry` so a threshold can be tied to the golden set that
  produced it, but changing the golden set does not require rebuilding the
  index.

### Canonicalization

The hash input is a JSON object serialized with:

- Sorted keys (`sort_keys=True`).
- No insignificant whitespace (`separators=(",", ":")`).
- UTF-8 encoding.
- No trailing newline.

This makes the hash stable across Python versions and across different
implementations of the same logical input.

### Collision handling

After computing an 8-character prefix, the builder checks whether
`index_registry` already contains a row with that `index_version`. If it does
**and** the recorded hash inputs differ (a real collision, not a
rebuild-of-the-same-index), the builder extends the prefix to 12 characters
and rechecks. If a 12-character prefix also collides, the builder fails with
a clear error rather than silently reusing or overwriting a row.

An 8-hex prefix has 2^32 values. At the scale this project targets (a few
hundred indexes over the project's lifetime), a collision is astronomically
unlikely. The check exists because "unlikely" is not "impossible", and
because the cost of the check is one indexed lookup.

**Rebuild of the same index.** If the 8-character prefix already exists and
the recorded hash inputs are identical, the builder treats it as a duplicate
build: it does not create a new row, and the caller is expected to skip the
build (see M2.3 idempotency). This is what makes `make index-build` safe to
re-run.

### Where `pgvector_version` is captured

At build time, from `SELECT extversion FROM pg_extension WHERE extname =
'vector'`. The value is stored in `index_registry.pgvector_version` and is
**not** re-queried at query time. If the extension is upgraded after the
index is built, the registry row continues to name the version that built
it; a subsequent build (which would produce a different `index_version`)
is required to make the upgrade visible in the identifier. This is the
correct behavior: the identity describes the index as built, not the
cluster as currently configured.

## Alternatives considered

| Option | Why not |
|---|---|
| **Sequential ids (`idx-0001`, `idx-0002`)** | Human-friendly but not content-addressed. Two builds from the same inputs produce different ids, which makes "did anything change?" unanswerable from the id alone. It also breaks idempotency: the builder cannot skip a rebuild without a separate content-comparison step. |
| **Full 64-hex SHA256** | Content-addressed and collision-free in practice, but 64 characters in a Prometheus label is a cardinality and readability cost with no benefit at this scale. |
| **UUID per build** | Not content-addressed. Same problem as sequential ids. Also longer than needed. |
| **Timestamp-based id** | Not content-addressed, and reveals wall-clock time in a label that has nothing to do with time. Two builds in the same second collide unless a suffix is added. |
| **8-hex prefix without collision check** | Cheapest, and "works" until it does not. A collision silently overwrites a registry row or mis-associates embeddings with the wrong index — a class of bug that is nearly impossible to diagnose in production. The check is one indexed lookup; skipping it is not worth the risk. |
| **Include `hnsw_ef_search` in the hash** | Conflates "the index" with "how the index was queried". Changing `ef_search` is a knob turn, not a rebuild. Including it would force a rebuild every time the operating point is adjusted. |
| **Include `golden_set_version` in the hash** | Same confusion: the golden set is evaluation metadata, not a build input. Including it would tie index identity to the evaluation harness. |

## Consequences

**Positive**

- Two builds from the same inputs produce the same `index_version`. This is
  what makes `make index-build` idempotent and what makes blue/green a
  pointer swap rather than a data migration.
- Two builds from different inputs — including ones that change only
  `metric` or only `preprocessing_version` — produce different ids. A change
  that would silently return different neighbours is visible in the label
  before anyone runs a query.
- The 12-character format is readable in logs and cheap as a Prometheus
  label.
- `pgvector_version` is captured once, at build, and stored. Upgrading the
  extension does not retroactively relabel an index that was built by an
  older version.

**Negative / accepted trade-offs**

- **The hash is opaque.** A human reading `idx-a3f9e021` cannot tell which
  model or catalog it corresponds to without looking it up in the registry.
  Mitigated by the registry row containing every component; the id is for
  correlation, not for inspection.
- **Collision handling is more code than not having it.** Truncated to 8
  hex, the probability of collision across a few hundred builds is on the
  order of 10^-5. The check is unconditional and cheap; the cost of getting
  it wrong is high.
- **Changing any build parameter invalidates the id.** This is the point,
  but it means an operator who changes `hnsw_m` "just to see" must publish
  the new id (and rebuild the evaluation report) rather than reusing the old
  one. The registry makes the change visible; the discipline is on the
  operator.
- **Extension upgrades are invisible until a rebuild.** An operator who
  upgrades pgvector and expects existing indexes to be relabeled will be
  surprised. This is deliberate: the id describes the index as built. The
  operational answer is documented in `docs/runbook.md`: after an extension
  upgrade, rebuild if the upgrade is expected to affect recall.

## References

- `docs/adr/0006-pgvector-schema.md` — the schema containing
  `index_registry` and the fields this ADR hashes over
- `docs/retrieval-and-evaluation.md` — the M2 design doc, which references
  this ADR for the id format
- `docs/contracts.md` § 2.1 — `meta.index_version` in the recommend
  response
- `docs/contracts.md` § 4.1 — `recsys_active_index_info{index_version=...}`
  metric
- M1 `docs/embedding-pipeline.md` § 3 — the analogous identity patterns for
  `model_version` and `catalog_snapshot`
