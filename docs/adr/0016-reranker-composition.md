# ADR-0016: Re-ranker composition

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

The retrieval path returns the top-k items by cosine similarity. That
ordering answers "which items are nearest in embedding space", which is
not the same as "which items should be shown first". Three signals the
model does not encode are worth blending in:

- **Popularity** — an item everyone interacts with is a safer
  recommendation than an equally similar item nobody touches.
- **Recency** — a newly added item has not had time to accumulate
  interactions, so popularity under-weights it; a decay term corrects
  the bias without ignoring the signal.
- **Diversity** — a list of ten near-identical items wastes the user's
  attention. MMR trades a little relevance for coverage.

The M0 scaffold names a formula (`README.md` § Retrieval, re-ranking)
and defers the re-ranker to M3 (`docs/adr/0005-m2-scope-retrieval-only.md`).
The empty `Reranker` protocol in `src/recsys/retrieval/rerank.py` is the
plug-in point. This ADR fixes what the re-ranker actually computes, the
composition rules, the failure behavior, and how it is measured.

A first-draft formula — `w_sim·sim + w_pop·pop + w_rec·rec` — was
rejected during review. The three terms have different ranges:

- `sim` is a cosine similarity, in `[-1, 1]` (in practice `[0, 1]` for
  sentence embeddings).
- `pop` is an unbounded non-negative number whose distribution depends
  entirely on the popularity provider (in M3, a synthetic score in
  `[0, 1000)`; in M5, an event count).
- `rec` is a decay in `[0, 1]`.

Adding them without normalization lets `pop` dominate: a single item
with a score of 900 contributes 180 to a weighted sum where `sim = 1.0`
contributes 0.7. The re-ranker would produce a popularity list with a
similarity tie-break, not a blended ranking.

## Decision

### Composition, in order

The re-ranker runs three steps in a fixed order:

1. **Normalize** each signal within the candidate window.
2. **Blend** the normalized signals with configurable weights.
3. **Diversify** with MMR on the top-`mmr_window` of the blended list.

The order matters. Normalizing after blending is meaningless (the sum is
already scaled by the weights). Diversifying before blending would make
MMR operate on a signal that does not yet include popularity or recency,
which is not what "diversity" is meant to be measured against.

### Step 1 — Normalization

**`sim`:** min-max scaled within the candidate window.

```
sim_norm = (sim - min_sim) / (max_sim - min_sim)
```

The window is the retrieval result, not the catalog. Retrieval already
chose the top candidates; re-ranking orders them. Min-max over the window
is the right scale because the question is "which of these candidates
is most similar", not "how similar is this candidate in absolute terms".

**`pop`:** log-scaled, then min-max.

```
pop_log = log1p(max(pop, 0))
pop_norm = (pop_log - min_pop_log) / (max_pop_log - min_pop_log)
```

`log1p` compresses the heavy tail. A popularity distribution is
long-tailed: the top item may have 10 000 interactions and the fiftieth
may have 50. Without the log, the first item dominates and the rest are
indistinguishable. With it, the ordering is preserved but the spacing
between adjacent candidates is meaningful.

**`rec`:** a time-decay, already in `[0, 1]`.

```
rec_norm = 0.5 ** (age_days / half_life_days)
```

`half_life_days` is config (`RERANK_RECENCY_HALF_LIFE_DAYS`, default 90).
An item added today has `rec_norm = 1`; an item added 90 days ago has
`0.5`; an item added 270 days ago has `0.125`. The exponential form has
one parameter and does not reach zero, so an old item is deprioritized
but never excluded.

**Degenerate cases.** If `max == min` for a signal (every candidate has
the same value), `_norm` returns `0.5` for every candidate. This is the
neutral choice: the signal contributes the same amount to every blended
score and therefore does not change the ordering. Returning `0.0`
instead would silently remove the signal's weight from the sum; returning
`1.0` would give it full weight for no information.

**Where normalization windows come from.** The candidate window is the
retrieval result, which the request path fetches with
`candidate_k = 4 × k` (see "Candidate window" below). Every signal is
normalized over that same window, so the four values a caller sees for a
given item are comparable to each other.

### Step 2 — Blend

```
score = w_sim * sim_norm + w_pop * pop_norm + w_rec * rec_norm
```

Weights are config (`RERANK_W_SIM`, `RERANK_W_POP`, `RERANK_W_REC`),
default `0.7 / 0.2 / 0.1`. They are validated at startup:

- Each weight in `[0.0, 1.0]`.
- The sum is `> 0`. It is not required to be exactly `1.0`: a caller may
  want to scale the whole score without changing the ratios, and forcing
  a sum of one would make that impossible. A weight of zero is allowed
  and means "this signal is not consulted", which is how an experiment
  arm compares "retrieval plus popularity" against "retrieval plus
  popularity plus recency" without a code change.

The default weights are a starting point, not a derivation. The
evaluation table (`README.md` § Offline evaluation) has a row for
retrieval-with-re-ranker; a weight change is an experiment arm (ADR-0017),
not a config edit that silently changes production behavior.

### Step 3 — Diversify with MMR

MMR is applied to the top-`mmr_window` of the blended list, not the full
candidate window. `mmr_window` is config (`RERANK_MMR_WINDOW`, default
50).

MMR is O(W²) in the window size W: it computes pairwise similarity
between the selected items and every remaining candidate at each step.
At W=400 (a `candidate_k = 4 × k` for `k = 100`) that is 160 000
similarity computations per request on the hot path. At W=50 it is
2 500, which is bounded and cheap. The 50-item window is the part of the
list a user sees (or sees after one "show more"); ordering beyond it does
not change what they see.

The MMR score for a candidate at each selection step is:

```
mmr(c) = lambda * blended(c) - (1 - lambda) * max_similarity(c, selected)
```

`lambda` is config (`RERANK_MMR_LAMBDA`, default 0.7). `lambda = 1.0`
disables diversity (the result is the blended list); `lambda = 0.0`
ignores relevance. The default is 70% relevance, 30% diversity.

**Pairwise similarity is computed on the embeddings the retrieval layer
already has.** The candidate vectors are the same ones the ANN search
returned; computing cosine between them is one matrix operation on the
window, not an extra database round trip. If the retrieval layer does
not expose the vectors, MMR is skipped (see "Failure behavior" below),
not recomputed from a separate source.

**MMR is only applied when `k >= min_k_for_mmr`.** Config
(`RERANK_MMR_MIN_K`, default 10). For a top-5 list, diversity has no room
to operate: dropping a near-duplicate for a tenth-ranked item makes the
list worse, not better. The threshold is where MMR starts helping.

### Candidate window

The request path fetches `candidate_k = k × candidate_multiplier`
neighbours from the backend, where `candidate_multiplier` is config
(`RERANK_CANDIDATE_MULTIPLIER`, default 4).

The multiplier is the slack that lets re-ranking improve the answer. If
the backend returns exactly `k` items, the re-ranker has no candidates to
promote: it can only reorder the same set, which is what the blend does
without MMR's ability to reach for a tenth-ranked item. A multiplier of 4
means a `k = 10` request retrieves 40 candidates; the re-ranker picks the
best 10 by the blended signal, then MMR diversifies the top-50 (which,
for `k = 10`, is the whole 40-candidate window).

**The multiplier has a cost.** Retrieval latency scales with `candidate_k`,
not `k`: the backend's ANN search returns 40 rows instead of 10. On the
pgvector backend the difference is small (the HNSW graph walk is the
dominant cost, not the row count), but on a very large catalog the extra
30 rows add up. M4's load test measures the actual cost; if the multiplier
proves expensive, it is config and can be lowered for the `recommend`
class without a code change.

### Failure behavior

**A re-ranker that raises is skipped, and the retrieval result is
returned.** The response carries `meta.source = "ann"` (the retrieval
source) and `meta.rerank` is either absent or lists the step that failed.
The request succeeds with a lower-quality answer; it does not become a
500.

The reason: re-ranking is a refinement, not a correctness requirement.
The retrieval path already returned a valid `k`-item list; the re-ranker
orders it better. A failure in the refinement must not fail the request.

**A signal provider that raises is treated as neutral.** If the
popularity provider (M5) is unavailable, `pop_norm` is `0.5` for every
candidate, and the blend reduces to `w_sim * sim_norm + w_rec * rec_norm`.
The weights are unchanged; the missing signal simply does not
differentiate candidates. This is the same convention as the "max == min"
degenerate case, which keeps one code path for "no information".

**MMR is skipped when the retrieval layer does not expose vectors.**
Some backends (a remote service that returns ids and scores only, a
future backend with a different protocol) may not hand back the raw
vectors. In that case MMR is not applied, the blended list is returned,
and a `rerank.mmr.skipped` log line records why. The alternative —
fetching vectors from a second source just to run MMR — adds a round
trip for a step that is optional by design.

### Observability

New metrics:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `recsys_rerank_duration_seconds` | histogram | `arm` | Time spent in the re-ranker, by experiment arm |
| `recsys_rerank_failures_total` | counter | `step`, `type` | A step (`normalize`, `blend`, `mmr`) that raised |
| `recsys_rerank_skipped_total` | counter | `reason` | `reason` ∈ `no_vectors` / `k_below_threshold` |
| `recsys_rerank_signal_missing_total` | counter | `signal` | A provider that raised and was treated as neutral |

The re-ranker's config (weights, `lambda`, half-life, window sizes) is
recorded in the response's `meta.rerank` object so a client can tell
which configuration produced a list, and in the log line for the request
(`rerank.config` at DEBUG). This is what makes an experiment arm
auditable after the fact: the same query under two arms produces two
responses whose `meta.rerank` differ.

### The evaluation table

M3 fills the `pgvector HNSW + re-ranker` row of the evaluation table
(`README.md` § Offline evaluation). The row is measured with the same
golden set and the same metrics as the retrieval-only row, so the delta
is attributable to the re-ranker and nothing else.

**The re-ranker must not lower NDCG@10.** If it does, the weights or the
MMR λ are wrong, and the row is a regression, not a feature. The
threshold for the row is set after the first measurement (ADR-0010), the
same procedure as every other system.

The evaluation uses the same re-ranker the API runs, with the same
config. A separate "evaluation re-ranker" would be a second
implementation to keep in sync; the config is the seam.

## Alternatives considered

| Option | Why not |
|---|---|
| **Raw weighted sum without normalization** | `pop` is unbounded and `sim` is in `[-1, 1]`; the sum is a popularity ranking with a similarity tie-break. The review that rejected this is recorded in the ADR's Context. |
| **Z-score normalization instead of min-max** | Z-score assumes a roughly Gaussian distribution. `pop` is heavy-tailed and `sim` is bounded; min-max (with log for `pop`) is the right shape for both. |
| **Rank-based normalization (each signal becomes its rank)** | Discards the magnitude differences within a window: the difference between rank 1 and rank 2 is the same as between rank 40 and rank 41. For `sim`, where the top three items may be near-identical, that loses information the blend should use. |
| **MMR on the full candidate window** | O(W²) on the hot path. At W=400 it is 160 000 similarity computations per request for candidates the user will not see. The 50-item window is the part that matters. |
| **MMR on the top-k only (no extra window)** | Defeats the point: MMR's job is to reach for a lower-ranked, less-similar item in place of a near-duplicate. With only `k` candidates there is nothing to reach for. |
| **Re-rank the retrieval result in place (no candidate multiplier)** | Same problem: the re-ranker can only reorder `k` items, which the blend already does. The multiplier is the slack the re-ranker needs to improve the answer. |
| **A re-ranker failure returns 500** | Turns a refinement failure into a request failure. The retrieval result is already valid; the request should succeed with it. |
| **Cache the re-ranked result** | The recency term is a function of wall-clock time, so the same retrieval inputs produce different outputs at 10:00 and 11:00. Caching the re-ranked list serves a stale ordering; ADR-0015 caches retrieval, not the re-ranked result. |
| **A learned re-ranker (a small model)** | Needs training data the project does not have at M3 (the event log lands in M5). The rule-based composition is a baseline that is measurable and explainable; a learned model is a candidate for a future ADR once there is data to train it on. |
| **`lambda` per experiment arm in the query string** | The arm belongs to the assignment (ADR-0017), not to the request. A client that could set `lambda` could compare arms against each other and invalidate the experiment. |
| **Different re-rankers for `recommend` and `similar`** | `similar` returns items similar to a seed item, where popularity and recency are less meaningful (the user asked for neighbors of this item, not for something popular). M3 uses the same re-ranker for both for simplicity; if `similar` measurements show the blend hurts it, a future ADR splits them. |

## Consequences

**Positive**

- **The blend is dimensionally sensible.** Every signal is in `[0, 1]`
  after normalization, so a weight of 0.2 contributes 0.2 at most. The
  default `0.7 / 0.2 / 0.1` reads as the ranking it produces.
- **The candidate multiplier is what makes re-ranking meaningful.**
  Without it, the re-ranker reorders the same `k` items; with it, the
  blend can promote an item retrieval ranked 25th into the top 10.
- **MMR cost is bounded.** 50-item window, not 400: O(50²) = 2 500
  similarity computations per request at most.
- **A re-ranker bug degrades quality, not availability.** The retrieval
  result is the floor; a failed refinement does not fail the request.
- **The response names the arm.** `meta.rerank` carries the config, so
  the same query under two arms produces two auditable responses.

**Negative / accepted trade-offs**

- **The candidate multiplier increases retrieval cost by ~4×.** Retrieval
  returns 40 rows for a `k = 10` request. Mitigated by measuring the
  actual cost in M4; if the multiplier is expensive, it is config.
- **Normalization is per-window, not per-catalog.** An item's
  `pop_norm` depends on which other candidates are in the window. Two
  requests with different `k` see the same item normalized differently.
  This is correct (the blend compares candidates within a request) but
  it means the blended score is not comparable across requests.
- **Min-max is sensitive to outliers.** A single candidate with a
  popularity far above the rest compresses every other candidate's
  `pop_norm` toward zero. The `log1p` step mitigates this; a robust
  alternative (median absolute deviation) is a future change if a
  measurement shows the compression hurts.
- **MMR can lower NDCG@10.** By design: it trades relevance for
  diversity. The evaluation table records the effect; a threshold that
  is lower than the retrieval-only row's would be a problem, but a small
  decrease is expected and documented.
- **MMR is skipped when vectors are not available.** A backend that
  does not expose vectors gets less diversity. This is a capability
  boundary, not a bug; a future backend that wants MMR exposes the
  vectors.
- **The weight defaults are a judgment.** 0.7 / 0.2 / 0.1 is a
  reasonable start, not a tuned result. M4 tunes them against measured
  NDCG; the values are config and the change is an experiment arm, not a
  silent edit.
- **MMR's `lambda` interacts with the candidate multiplier.** A larger
  multiplier gives MMR more to reach for, which changes the optimal
  λ. The two are configured independently; the interaction is a
  measurement in M4, not a value in this ADR.

## References

- `docs/adr/0005-m2-scope-retrieval-only.md` — why the re-ranker was
  deferred to M3 and why a protocol stub exists now
- `docs/adr/0009-golden-set-and-metrics.md` — the metrics the re-ranker
  is measured against
- `docs/adr/0015-cache-and-circuit-breaker.md` — the cache stores
  retrieval, not the re-ranked result
- `docs/adr/0017-experiment-assignment.md` — weights are an experiment
  arm
- `src/recsys/retrieval/rerank.py` — the protocol this ADR implements
- `tests/unit/test_rerank.py` and
  `tests/integration/test_recommend_with_rerank.py`
