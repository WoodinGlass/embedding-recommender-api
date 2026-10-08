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

Each signal is normalized to `[0, 1]` **within the candidate window**.
The window is the retrieval result (`candidate_k = k × candidate_multiplier`),
not the catalog. The question at this stage is "which of these candidates
is best on this signal", not "how good is this candidate in absolute
terms".

**`sim`:** min-max scaled within the window.

```
sim_norm = (sim - min_sim) / (max_sim - min_sim + eps)
```

`eps = 1e-8`. The candidate window comes from an ANN search, whose
result set is usually tight; when every candidate has the same
similarity (a degenerate window), `max == min` and the divisor collapses
to `eps`, which drives every `sim_norm` to `0.0`. The degenerate-case
rule below overrides that outcome before it can be observed: if
`max == min`, every candidate gets `0.5` regardless of the raw formula.

`Candidate.similarity` is always a **cosine similarity in `[-1, 1]`**,
higher is better. Conversion from a distance (the `PgvectorBackend`
returns `vector_cosine_ops` distances, where `distance = 1 - similarity`)
happens **in the caller**, before the candidate is constructed. The
re-ranker never guesses at the metric; it reads the field and treats
it as similarity.

**`pop`:** rank-based, higher-is-better.

```
rank_i = position of candidate i in the window sorted by popularity DESC (0-based)
pop_norm = 1.0 if n == 1 else 1.0 - (rank_i / (n - 1))
```

The highest-popularity candidate has `rank_i = 0` and gets `pop_norm =
1.0`; the lowest gets `pop_norm = 0.0`. The convention matches `sim`
and `rec`: a larger value is a better candidate on that signal, and the
blend multiplies each by a positive weight and sums. There is no
inversion step in the blend.

**Why not store the raw rank (`rank_i / (n - 1)`, where 0 is best) and
invert in the blend.** An earlier draft did this and it was
unnecessarily confusing: two conventions for the same three signals
("higher is better" for `sim` and `rec`, "lower is better" for `pop`)
is one place for a future reader to get the polarity wrong, and the
"invert in the blend" step has no behavioral benefit. The tie-break
argument that motivated the draft was wrong: two candidates with equal
popularity produce the same `pop_norm` under either convention, so the
final `(score DESC, item_id ASC)` sort (see "Determinism" below) is
what resolves the tie, and it does so identically in both cases.

**Why rank-based and not `log1p` + min-max.** An earlier draft used
`log1p(pop)` followed by min-max. Review rejected it: the log compresses
the tail, but the min-max on top of the log is still sensitive to a
single outlier at the top of the window. A single item with `pop = 10^6`
in a window whose second-highest is `pop = 100` compresses every other
`pop_norm` toward `0.0` after the log, so the blend sees "one item with
popularity, everyone else tied at zero". Rank-based normalization
preserves the ordering without compressing the spacing between the
non-outlier candidates: candidate ranks 2 through `n` remain evenly
spaced regardless of how large the outlier is. The `log1p` step is
discarded; `pop` is rank-based, `sim` is min-max, and the two methods
are chosen for the distribution each signal actually has.

**`rec`:** a time-decay, already in `[0, 1]`.

```
rec_norm = 0.5 ** (age_days / half_life_days)
```

`half_life_days` is config (`RERANK_RECENCY_HALF_LIFE_DAYS`, default 90).
An item added today has `rec_norm = 1`; an item added 90 days ago has
`0.5`; an item added 270 days ago has `0.125`. The exponential form has
one parameter and does not reach zero, so an old item is deprioritized
but never excluded.

`age_days` is measured from `item.created_at`, not `item.updated_at`.
The choice is deliberate: `updated_at` moves when metadata changes
(a description edit, a category re-tag), which is not the same as "the
item is new". A future ADR may add a second decay term for freshness of
metadata if a measurement shows it matters; at M3 the signal is "how
recently was the item introduced to the catalog".

**Degenerate cases.** If `max == min` for a signal (every candidate has
the same value), `_norm` returns `0.5` for every candidate. This is the
neutral choice: the signal contributes the same amount to every blended
score and therefore does not change the ordering. Returning `0.0`
instead would silently remove the signal's weight from the sum; returning
`1.0` would give it full weight for no information. A window of one
candidate is degenerate by definition and follows the same rule.

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

**MMR is enabled by an explicit flag, not by the presence of vectors.**
`RerankConfig.enable_mmr: bool` controls whether MMR runs. When it is
`False`, the MMR step is skipped and the blended list is returned
unchanged; `meta.rerank.mmr_active = False`. When it is `True` and the
candidates carry no vector (a backend that does not expose them), the
step is skipped with `recsys_rerank_skipped_total{reason="no_vectors"}`
and `meta.rerank.mmr_active = False`.

An earlier draft inferred MMR from `Candidate.vector is not None`. It
was rejected: the effective configuration would then depend on which
backend happened to serve the request, and two responses produced under
the same arm would silently differ in behavior. The explicit flag makes
`meta.rerank.mmr_active` auditable and keeps an experiment arm honest.

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

**A signal provider that times out is treated the same as one that
raises.** Both providers (`PopularityProvider`, `RecencyProvider`) accept
a per-call timeout. A timeout is a failure: the signal is neutral, the
request proceeds, and `recsys_rerank_signal_missing_total{signal}` is
incremented. A slow provider must not block the request for longer than
the timeout, and must not fail the request when it does.

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
| `recsys_rerank_signal_missing_total` | counter | `signal` | A provider that raised or timed out and was treated as neutral |
| `recsys_rerank_mmr_active_total` | counter | `active` | `active` ∈ `true` / `false`; MMR ran or was skipped on this request |

The re-ranker's config (weights, `lambda`, half-life, window sizes) is
recorded in the response's `meta.rerank` object so a client can tell
which configuration produced a list, and in the log line for the request
(`rerank.config` at DEBUG). This is what makes an experiment arm
auditable after the fact: the same query under two arms produces two
responses whose `meta.rerank` differ.

**Cardinality budget.** The label values below are the entire set the
project accepts for the re-ranker metrics:

| Metric | Allowed labels | Max cardinality |
|---|---|---|
| `recsys_rerank_duration_seconds` | `arm` ∈ {`retrieval`, `blend`, `blend_mmr`} | 3 |
| `recsys_rerank_failures_total` | `step` ∈ {`normalize`, `blend`, `mmr`}, `type` (bounded to top 5 exception classes) | 15 |
| `recsys_rerank_skipped_total` | `reason` ∈ {`no_vectors`, `k_below_min`, `no_candidates`} | 3 |
| `recsys_rerank_signal_missing_total` | `signal` ∈ {`popularity`, `recency`} | 2 |
| `recsys_rerank_mmr_active_total` | `active` ∈ {`true`, `false`} | 2 |

`item_id`, `seed_item_id`, `category`, `brand`, `language`, and any
per-request value are **forbidden** as labels. Per-request identity
belongs in logs, traces, or exemplars.

### The evaluation table

M3 fills the re-ranker rows of the evaluation table (`README.md` §
Offline evaluation). The evaluation runs **three arms** on the same
golden set:

- **`retrieval`** — the retrieval result, no re-ranking. The floor.
- **`blend`** — retrieval, normalize, weighted blend. No MMR.
- **`blend_mmr`** — the blend, then MMR on the top-`mmr_window`.
  Requires a backend that exposes candidate vectors.

Three arms, not one, because "the re-ranker" is not a single change:
the blend and MMR are independent steps, and the evaluation has to
attribute a delta to each. A single "with re-ranker" row hides which
step moved the metric. If the blend helps and MMR hurts, the row would
show the net effect and the reader would have no way to act on it.

The three arms share the same golden set, the same re-ranker
implementation, and the same config file. Only `enable_mmr` differs
between `blend` and `blend_mmr`, and only the re-rank step is skipped
between `retrieval` and `blend`. This is what makes the deltas
attributable.

**Production default is `blend` until MMR is measured to help.**
`blend_mmr` runs in the evaluation only. Once a measurement shows the
`blend_mmr` row is not below the `blend` row on NDCG@10 and the latency
cost (M4) is acceptable, the production default flips. Flipping it
earlier would ship a step whose effect has not been measured.

**The re-ranker must not lower NDCG@10 below the retrieval arm.** If
`blend` is below `retrieval`, the weights are wrong. The `blend_mmr`
row is allowed a small decrease against `blend`: MMR trades relevance
for diversity by design, and the size of the trade is a number the
evaluation records, not a value this ADR fixes.

Thresholds for each arm are set after the first measurement (ADR-0010),
the same procedure as every other system. The re-ranker used in
evaluation is the same implementation the API runs, with the same
config. A separate "evaluation re-ranker" would be a second
implementation to keep in sync; `enable_mmr` and the weights file are
the seams.

### Determinism

The re-ranker is a pure function of its inputs. Given the same
`Candidate` list and the same `RerankConfig`, it produces the same
output list **byte-for-byte**, including tie order. This is required by
three downstream uses:

- The M2 evaluation gate compares reports across CI runs and would be
  flaky if tie ordering drifted.
- An A/B assignment (M5) is meaningful only if a variant's behavior is
  reproducible; a re-ranker whose order depended on hash randomization
  would put two callers of the same arm in different buckets.
- A regression (or its absence) after a deploy is diagnosed by
  comparing outputs; a non-deterministic re-ranker makes "the output
  changed" and "the output is different this run" indistinguishable.

**The rule.** Every sort in the re-ranker uses a total order:
`(score DESC, item_id ASC)`. The score is the blended (or MMR) score;
the tie-break is the item id, ascending, byte-for-byte. Two candidates
with equal scores are ordered by id, not by their position in the input
list and not by their insertion order into a dict.

**Providers must be deterministic too.** A popularity provider that
returns a different value for the same item across two calls in one
process is a bug. The synthetic provider in M3 is a hash of the item
id; the event-based provider in M5 must read a stable snapshot, not a
counter that increments per call. The re-ranker cannot enforce this;
the provider contract states it, and a test asserts that the same input
produces the same output.

### Config validation

The re-ranker's config is read from `config/hot.yaml` (ADR-0022) at
startup and validated before the first request. A malformed config is a
startup failure, not a warning: a re-ranker with a negative weight
produces nonsense scores that are worse than not re-ranking at all, and
silently falling back to "no re-ranking" would hide the misconfiguration
behind a metric that only moves by a few points.

The rules, all enforced at startup:

- `w_sim`, `w_pop`, `w_rec` are each in `[0.0, 1.0]`.
- `w_sim + w_pop + w_rec == 1.0` within `1e-6`. A sum of zero would
  make every blended score zero; a sum other than one is a likely typo
  that changes the meaning of every weight.
- `mmr_lambda` is in `[0.0, 1.0]`.
- `recency_half_life_days > 0`.
- `candidate_multiplier >= 1`.
- `mmr_window >= 1` and `mmr_min_k >= 1`.
- `enable_mmr` is a bool.

A change to any of these values is an experiment arm (ADR-0017), not a
silent config edit. The values that produced a response are echoed in
`meta.rerank` so a client can tell which configuration it saw.

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
| **`log1p` + min-max for popularity** | An earlier draft. Review rejected it: the min-max on top of the log is still sensitive to a single outlier, which compresses every other candidate toward zero. Rank-based normalization preserves the ordering of the non-outlier candidates regardless of the outlier. |
| **Infer MMR from `Candidate.vector is not None`** | Makes the effective configuration depend on which backend served the request, so two responses under the same arm silently differ. An explicit `enable_mmr` flag keeps `meta.rerank.mmr_active` auditable. |

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
