# ADR-0009: Golden set versioning, metrics, query encoding, and baselines

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

M2's exit criterion is that retrieval metrics are documented and enforced as
a CI gate. That implies four separate decisions, none of which are obvious:

1. **What is being measured against.** A metric without a stable ground truth
   is a number, not a measurement. If the golden set changes between runs,
   the threshold has no meaning.
2. **How retrieval quality is computed.** Recall@k, NDCG@k, and MRR each
   have multiple conventions; the choice must be fixed and documented.
3. **How a query is derived from seed items.** The API accepts
   `seed_item_ids: list[str]`. The retrieval layer needs a single vector.
   The transformation from a set of seed items to a query vector must be
   deterministic and reproducible.
4. **What retrieval is compared against.** A metric in isolation says
   nothing about whether retrieval is good or bad. Baselines give the metric
   meaning.

The M1 golden set (`evaluation/golden_set/queries.yaml`, 20 queries × 3
seeds × 7 relevant) already exists as the fast fixture for a future
evaluation gate. M2 formalizes the format, adds a version, defines the
metrics, and specifies the baselines that appear in the README table.

## Decision

### 1. Golden set format and versioning

**Format.** `evaluation/golden_set/v1.jsonl`. One JSON object per line:

```json
{"query_id": "q_science_fiction", "topic": "science_fiction", "seed_item_ids": ["i_0001", "i_0002", "i_0003"], "relevant_item_ids": ["i_0004", "i_0005", "i_0006", "i_0007", "i_0008", "i_0009", "i_0010"]}
```

Required fields:

- `query_id` (string, unique within the file)
- `topic` (string; informational, used for per-topic diagnostics)
- `seed_item_ids` (list of strings; 1..N, no duplicates, all in the catalog)
- `relevant_item_ids` (list of strings; disjoint from `seed_item_ids`, all
  in the catalog)

**Version.** The golden set is versioned by filename: `v1.jsonl`, `v2.jsonl`.
The version string `"v1"` is:

- declared in `evaluation/thresholds.yaml` as `golden_set_version: "v1"`;
- recorded in `index_registry.golden_set_version` when an index is built;
- written into every evaluation report.

**Rule:** an evaluation report is only comparable to another evaluation
report with the same `golden_set_version`. The CI gate fails if the report's
version does not match the threshold's version.

**Why JSONL and not YAML.** The M1 golden set is YAML for human
readability. As the set grows past a few dozen queries, JSONL is easier to
diff, to stream, and to validate line-by-line; each line is independently
parseable. The M1 YAML file will be superseded by `v1.jsonl` (same content,
different serialization).

**Why a filename version, not a field.** If the version were a field inside
the file, a query appended to `v1.jsonl` would not bump the version unless
the field were manually edited. A filename version makes the change
mechanically visible in `git diff` and in the threshold file.

### 2. Metric definitions

All metrics take a ranked list of retrieved item ids (`retrieved`) and a set
of relevant item ids (`relevant`), and evaluate the top `k` of the ranked
list.

**Recall@k**

```
recall_at_k = |set(retrieved[:k]) ∩ relevant| / min(|relevant|, k)
```

- Denominator `min(|relevant|, k)` is the correct convention when `|relevant|`
  may exceed `k`. If `|relevant| = 7` and `k = 10`, the denominator is 7; a
  system that returns all 7 in the top 10 has recall 1.0. If `|relevant| = 20`
  and `k = 10`, the denominator is 10; a system cannot recall more than 10
  relevant items in a top-10 list.
- Range: `[0, 1]`. Higher is better.

**NDCG@k (binary relevance)**

```
DCG@k   = Σ_{i=1..k} rel_i / log2(i + 1)
IDCG@k  = Σ_{i=1..min(|relevant|, k)} 1 / log2(i + 1)
ndcg@k  = DCG@k / IDCG@k    (0 if IDCG@k == 0)
```

- `rel_i` is 1 if `retrieved[i-1]` is in `relevant`, else 0. Binary
  relevance: an item is either relevant or it is not.
- `IDCG@k` is the DCG of the ideal ranking (all relevant items first), which
  is what a perfect system would score.
- Range: `[0, 1]`. Higher is better.

**MRR (mean reciprocal rank)**

```
mrr = 1 / rank(first relevant item)    (0 if no relevant item is retrieved)
```

- Evaluated over the full retrieved list, not just top `k`. In practice,
  the retrieval layer returns at most `max(k)` results, so MRR is computed
  against that list.
- Range: `[0, 1]`. Higher is better.

**Aggregation.** For a golden set with `Q` queries, each metric is the
arithmetic mean over queries. Per-topic breakdown is recorded in the report
for diagnostics but the threshold is applied to the aggregate.

**Binary vs graded relevance.** This ADR fixes binary relevance for M2. The
golden set (`relevant_item_ids`) is a set; membership is 0 or 1. Graded
relevance — where an item might be "strongly relevant" or "weakly relevant"
— requires a different golden set format and a modified NDCG formula. It is
**out of scope for M2** and deferred; if a future milestone wants it, a new
ADR and a new golden set version are required.

### 3. Query encoding

The API accepts a list of seed items. Retrieval needs a single vector.

**Encoding:** the query vector is the L2-normalized mean of the seed items'
embeddings.

```
v_q = normalize( (1/|S|) * Σ_{s ∈ S} e_s )
```

where `normalize(x) = x / ||x||_2` and `e_s` is the stored embedding of seed
item `s` under the index being queried.

**Why mean.** It is deterministic, requires no parameters, and works well
when seeds are drawn from a single topic cluster (which the sample catalog
and golden set are). It is the natural choice when no information about
relative importance of seeds is available.

**Why L2-normalize.** The metric is cosine distance (`vector_cosine_ops`),
which on L2-normalized vectors is a dot product. Stored embeddings are
L2-normalized by the M1 pipeline. The mean of normalized vectors is not
itself normalized unless all seeds are identical, so the query must be
re-normalized before it is compared. Omitting this step would produce
cosine distances that are scaled by `||v_q||` and would break the ordering
guarantee.

**Edge case: empty seed list.** The contract allows `seed_item_ids: []` for
a cold-start user. In that case, retrieval does not run — the caller falls
back to popularity (M3). The evaluation harness never produces an empty
seed list; the golden set requires at least one seed per query.

**Alternatives considered for query encoding, and why not.**

| Option | Why not |
|---|---|
| **Max-pooling across seeds** | Amplifies the dimensions where any seed has a high value; tends to produce a query vector dominated by outliers. Not order-stable for typical embeddings, and harder to justify without measurement. |
| **Weighted mean (recency, popularity)** | Requires weights that are not yet available (no event log until M5). Adding guessed weights would bake in an assumption the evaluation cannot validate. |
| **Concatenate seeds and encode the concatenation** | Requires re-encoding at query time with the model, which the M1 pipeline does not serve online. Would also make the query dependent on tokenization order. |
| **Use the first seed item only** | Throws away information and makes the golden set's 3 seeds meaningless. |
| **Centroid with per-seed re-weighting by cosine to the mean** | A refinement that could be measured in a future milestone. Not motivated by any observed failure in M2. |

The mean is the simplest choice that satisfies the determinism contract and
does not require parameters. If M2.4 measurements show a different encoding
would improve retrieval, that is a candidate for an experiment arm in M5,
not a silent change here.

### 4. Baselines

Every evaluation report includes the following baselines, computed against
the same golden set and the same `k`:

**Random baseline.** For each query, retrieve `k` items drawn uniformly
from the catalog, seeded by a deterministic function of the query id:

```
seed = int(sha256(query_id.encode()).hexdigest()[:16], 16)
```

so the same query always produces the same random items. This is the floor:
a system that does not beat random is worse than noise.

**Popularity baseline (synthetic).** For each query, score every catalog
item by a deterministic pseudo-score:

```
score(item_id) = int(sha256(item_id.encode()).hexdigest()[:8], 16) % 1000
```

take the top `k` by `(score desc, item_id asc)`.

This is **not real popularity.** It is a placeholder with the interface of a
popularity signal, so that:

- the evaluation harness can exercise a popularity-based retrieval path
  before M5 introduces a real event log;
- the README table has a mid-baseline against which a working embedding
  model can be compared.

The interface is:

```python
class PopularityProvider(Protocol):
    def scores_for(self, item_ids: Sequence[str]) -> dict[str, float]: ...
```

The synthetic implementation is registered as the default for M2. The M5
implementation will query the event log and be swapped in without changing
the evaluation harness.

**Determinism requirement.** Both baselines must be deterministic. A
non-deterministic baseline would make the evaluation report un-reproducible
and would make the gate unreliable (two runs of the same code could produce
different baseline scores). The implementations are unit-tested for
determinism.

**Exact kNN baseline.** For each query, compute the cosine distance from
the query vector to every item in the catalog, sort, and take the top `k`.
This is O(N) per query, which is why it is a baseline and not the
production path, but on the sample catalog (200 items) it is instant. It
gives the ceiling for retrieval: no ANN index can return better neighbours
than exact search on the same vectors.

**Production path (pgvector HNSW).** The actual retrieval under test.

The README table will have four rows:

| System | Recall@10 | NDCG@10 | MRR | ANN recall vs exact |
|---|---|---|---|---|
| Random baseline | … | … | … | n/a |
| Popularity (synthetic) | … | … | … | n/a |
| Exact kNN | … | … | … | 1.0 (by definition) |
| pgvector HNSW | … | … | … | … |

The exact kNN row is expected to have perfect ANN fidelity (its neighbors
*are* the exact neighbors) but the same Recall@k as its own retrieval
quality. This makes the table's structure self-documenting: the "ANN recall
vs exact" column measures how much HNSW gives up, not how good the vectors
are.

### 5. ANN fidelity metric

```
ann_recall_vs_exact = |top_k(HNSW) ∩ top_k(exact)| / k
```

per query, averaged over the golden set. This is what separates two failure
modes that would otherwise be indistinguishable:

- **Vector quality changed.** The model produces different vectors, so
  top-k changes. Both HNSW and exact see the new neighbors, and both
  improve or worsen together on Recall@k.
- **The index lost neighbors.** The vectors are the same, but HNSW returns
  a different top-k than exact search. Recall@k may look normal while the
  index is silently returning worse neighbors.

The first failure mode is a modeling concern; the second is an indexing
concern. The fidelity column attributes the difference to the right one.

ANN fidelity is computed against **exact kNN on the same vectors**, not
against the golden set. It is orthogonal to retrieval quality; a system with
poor vectors can still have perfect ANN fidelity.

## Alternatives considered

| Option | Why not |
|---|---|
| **No golden set versioning; one file, edited in place** | Thresholds would silently become invalid when the file changed. The gate would pass on a threshold computed for an old set of queries. |
| **Multiple small golden sets (one per topic)** | Adds bookkeeping and does not change what is measured. The single file with a `topic` field gives per-topic breakdown without fragmenting the artifact. |
| **Recall@k with denominator `|relevant|` (not `min(|relevant|, k)`)** | Underestimates recall when `|relevant| > k`, penalizing a system that returns all `k` slots full of relevant items. The `min` convention is standard and matches how the number is usually reported. |
| **NDCG@k with graded relevance** | Requires a richer golden set (per-item relevance scores) and human judgment to assign them. Binary relevance is objective and available; graded relevance is a future improvement, not a blocker for M2. |
| **Skip the random baseline** | Without a floor, a low score looks like failure when it might be indistinguishable from chance. The random baseline makes the floor visible. |
| **Skip the popularity baseline** | Without a mid-baseline, a system that beats random but loses to popularity looks acceptable. Popularity is a strong baseline on many recommendation tasks; it should be named. |
| **Use the golden set itself as the popular items** | Circular: relevant items would be popular by construction, inflating the baseline. The synthetic popularity uses item ids, not the golden set. |
| **Real popularity from a fake event log** | Adds a synthetic-data generator whose biases would be opaque. The synthetic score is honest about being a placeholder, and the interface is ready for real data in M5. |
| **Query encoding by re-encoding concatenated seed text** | Requires the online encoder path, which is M3. Also makes the query dependent on tokenization order, which the M1 pipeline does not guarantee for concatenated text. |
| **Compute ANN fidelity against the golden set** | Conflates indexing quality with retrieval quality. ANN fidelity should be a comparison between two methods on the same task, not a comparison of methods against an external ground truth. |

## Consequences

**Positive**

- **The threshold has meaning.** A threshold is only meaningful relative to
  a fixed golden set. Versioning makes that explicit: an evaluation report
  and a threshold are comparable if and only if their golden set versions
  match.
- **Metrics are unambiguous.** Recall@k, NDCG@k, and MRR each have a single
  definition, documented with the exact formula and the conventions
  (denominators, tie-breaking, empty cases).
- **Query encoding is deterministic.** Mean + L2-normalize produces the same
  query vector for the same seeds, in any process, on any machine. This is
  required for the evaluation report to be reproducible.
- **Baselines give the metric context.** "NDCG@10 = 0.85" is unreadable
  without "random = 0.15, popularity = 0.42, exact = 0.87". The table reads
  as a story about where retrieval stands.
- **ANN fidelity isolates a class of bug.** A drop in Recall@k caused by the
  index is distinguished from a drop caused by the vectors.

**Negative / accepted trade-offs**

- **Binary relevance limits NDCG's expressiveness.** A system that returns
  the most relevant items first scores no better than one that returns
  merely relevant items first. For the sample catalog, all items in a topic
  cluster are equally relevant by construction, so binary is the right
  granularity. A graded golden set would need human labeling and is deferred.
- **Synthetic popularity is a weak baseline.** It does not resemble real
  popularity; real popularity is concentrated, the synthetic score is not.
  A retrieval system could beat synthetic popularity without being useful
  on real data. Mitigated by: the README table is labeled clearly, and the
  interface is ready for M5.
- **Mean query encoding loses information.** When seeds span multiple
  topics, the mean is a compromise vector that may sit between clusters.
  Acceptable for the sample golden set, where seeds per query are drawn from
  one topic. A per-topic golden set would not benefit from a different
  encoding; a multi-topic query would, and that is a future improvement.
- **The exact kNN baseline does not scale.** On the sample catalog (200
  items) it is instant; on a 1M-item catalog it would be O(N) per query and
  unusable. This is deliberate: exact kNN is a baseline for the sample
  catalog, not a production alternative. If M2's README table is ever
  recomputed against a larger catalog, exact kNN moves to a subsampled
  probe set.
- **The popularity provider is a Protocol with one implementation.** A
  Protocol with a single implementation is boilerplate. It exists so that
  M5's swap does not touch the evaluation harness, which is the correct
  trade-off for a decision whose cost is small and whose future use is
  already planned.

## References

- `docs/retrieval-and-evaluation.md` — the M2 design doc
- `docs/contracts.md` § 2.1 — the request shape that defines `seed_item_ids`
- `evaluation/golden_set/v1.jsonl` — the golden set this ADR defines
- `evaluation/thresholds.yaml` — where `golden_set_version` is pinned
- `docs/adr/0006-pgvector-schema.md` — `index_registry.golden_set_version`
- `docs/adr/0010-evaluation-thresholds.md` — how thresholds are chosen and
  the absolute floor
- M1 `docs/embedding-pipeline.md` — the embedding pipeline whose output
  these metrics evaluate
- M5 milestone — where real popularity replaces the synthetic placeholder
