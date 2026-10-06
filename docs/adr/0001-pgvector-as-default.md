# ADR-0001: pgvector as the default vector store

- **Status:** Accepted
- **Date:** 2026-10-07
- **Deciders:** project maintainer

## Context

The retrieval layer must serve filtered approximate nearest neighbour (ANN)
queries over a catalog of 100k to 1M items, with a latency target of
p95 < 200 ms (`README.md` § Performance). Retrieval happens on the hot path of
`POST /v1/recommend`, alongside metadata filters that come from the request
(`category`, `brand`, `language` — see `docs/contracts.md` § 2.1).

Constraints that narrow the choice:

- Single-node deployment on one VPS or cloud instance (`docs/decisions.md` § 4).
  No dedicated vector database cluster, no separate ops team.
- Metadata (catalog, filters, experiment assignments, event log, churn
  features) already lives in PostgreSQL. A second datastore would introduce
  cross-store consistency questions the request path cannot afford.
- The service must degrade gracefully when the vector layer is unavailable
  (`docs/decisions.md` § 6): a fallback to popularity within the same filters
  is required, which implies the filter metadata must be queryable
  independently of the vector index.

## Decision

Use **PostgreSQL 16 with the pgvector extension**, HNSW index, cosine distance
on normalized embeddings, as the default vector store for retrieval.

FAISS remains available behind the `[bench]` extra as a benchmark baseline
(M2). A dedicated vector store (Milvus, Qdrant, Pinecone) is explicitly out of
scope; if it becomes necessary, that decision requires a new ADR.

## Alternatives considered

| Option | Why not |
|---|---|
| **FAISS in-process** | Fast and well-tuned, but there is no persistence story, no transactional filtering, and index updates require a full reload. Filtering must happen post-hoc, which breaks the "return k results after filters" guarantee at 1M items. Keeping it as a benchmark baseline gives us the throughput number without adopting its operational model. |
| **Qdrant / Milvus (dedicated vector DB)** | Purpose-built and faster under load, but adds a second stateful service, a second backup/restore story, and a second consistency boundary between the vector store and PostgreSQL metadata. The team is one person; the operational cost is real. |
| **Pinecone (managed)** | Removes operational burden, but adds a paid external dependency and a network hop on the hot path. It also removes the ability to run the whole project locally with `docker compose up`, which is a portfolio requirement. |
| **pgvector IVFFlat** | Simpler to build (no training step beyond the list count), but recall is more sensitive to data distribution and the recall/latency curve is worse for filtered queries. HNSW's cost is memory and build time, which are acceptable at this scale. |

## Consequences

**Positive**

- One datastore means filter + vector search happens in a single SQL query
  with transactional consistency between them. No cross-store reconciliation.
- Metadata (`category`, `brand`, `language`) and embeddings are updated in the
  same transaction, so an item can never be visible with a stale embedding.
- A single backup and restore procedure covers everything.
- Iterative index scans (pgvector 0.8+) handle selective filters without
  losing recall, which is the failure mode that makes in-process FAISS painful.
- `docker compose up` runs the entire stack locally with no external accounts.

**Negative / accepted trade-offs**

- Lower throughput ceiling than a dedicated vector engine. At the target scale
  the difference is absorbed by the Redis cache; the FAISS benchmark in M2 will
  quantify the gap so the trade-off is measured rather than asserted.
- HNSW index build time and memory footprint scale with catalog size. A
  1M-item rebuild is a batch operation, not an online one; the blue/green
  index swap in M2 exists precisely because of this.
- Fewer tuning knobs than a purpose-built engine. The relevant ones
  (`m`, `ef_construction`, `hnsw.ef_search`) are exposed as config
  (`docs/contracts.md` § 3) and their effect on the recall/latency curve is
  documented in M2.

## References

- `docs/decisions.md` § 3 (determinism), § 4 (storage tiers), § 6 (failure
  modes)
- `docs/contracts.md` § 3 (config enums, `INDEX_BACKEND`)
- `src/recsys/retrieval/registry.py` — the interface both backends implement
- FAISS benchmark: M2, results recorded in `README.md`
