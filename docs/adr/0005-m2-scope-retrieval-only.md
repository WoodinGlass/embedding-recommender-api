# ADR-0005: M2 scope is retrieval only; re-ranking moves to M3

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

The M2 exit criterion is "metrics are documented and enforced as a CI gate".
The milestone table in `README.md` originally listed re-ranking (popularity,
recency, diversity/MMR) alongside retrieval as M2 scope. That combination is
problematic for two reasons.

First, **attribution**. If retrieval and re-ranking ship together, a change
in NDCG@10 cannot be attributed to either. The retrieval change and the
re-ranking change are confounded. The M2 report would say "NDCG improved"
without being able to say why, and the M3 A/B experiments would inherit an
unknown baseline.

Second, **ordering**. Re-ranking transforms the output of retrieval. Tuning
a re-ranker against a retrieval layer whose recall/latency trade-off is
itself still being settled in M2 means tuning against a moving target. A
re-ranker is useful once retrieval is stable, not while it is being built.

## Decision

**M2 delivers retrieval and offline evaluation, and nothing else.**

- Retrieval: a `Backend` protocol, a `NumpyBackend` (exact kNN, for local
  development and as the ANN-fidelity baseline), and a `PgvectorBackend`
  (HNSW, production).
- Evaluation: Recall@k, NDCG@k, MRR, ANN fidelity vs exact kNN, three
  baselines (random, synthetic popularity, exact kNN), a versioned golden
  set, and a threshold gate that runs in CI.
- **No re-ranker.** Re-ranking (popularity/recency/MMR) ships in M3,
  alongside the serving API, where it becomes an experiment arm.

To avoid a large refactor when M3 lands, we add an empty `Reranker` protocol
now (`src/recsys/retrieval/rerank.py`) with a single method and no
implementations. The retrieval layer does not call it in M2. M3 fills in
implementations and wires the call site; the interface itself does not move.

`README.md` is updated so feature #3 (re-ranking) is described as an M3
deliverable. Documenting a feature as shipping in M2 while deciding it
does not is a documentation bug, and documentation bugs are indistinguishable
from real bugs to a reader.

## Alternatives considered

| Option | Why not |
|---|---|
| **Ship re-ranker in M2 anyway** | Confounds the M2 metrics. The published table would mix retrieval and re-ranking effects and would not be reproducible in M3 when the re-ranker is exercised as an arm. Attribution is worth more than the saved milestone. |
| **Ship re-ranker in M2 but evaluate separately** | Requires running two evaluations (retrieval-only and retrieval+rerank) and publishing both. That is the right output for M3, not M2. Doing it in M2 adds work without adding a decision: the re-ranker's weights are not yet chosen, and any choice would be a guess. |
| **Defer re-ranker indefinitely** | It is a listed feature of the project (README feature #3) and a lever for A/B testing (M5). Deferring it past M3 removes a planned experiment arm. Not acceptable. |
| **Stub the re-ranker interface later, in M3** | The stub is trivial to add now and expensive to retrofit later: adding a protocol to an already-used module means a migration of call sites. Adding it now costs a dozen lines and no runtime behaviour. |

## Consequences

**Positive**

- **Clean attribution.** The M2 evaluation table is retrieval-only. When M3
  adds re-ranking, the delta between M2 and M3 tables is exactly the
  re-ranker's effect, measured against a fixed baseline.
- **Stable tuning target.** HNSW parameters and filter strategy are settled
  in M2 before anything downstream depends on them.
- **Cheap extensibility.** The `Reranker` protocol stub costs ~15 lines and
  a docstring, and gives M3 a defined place to plug in. No call sites to
  migrate.
- **Honest documentation.** README feature #3 moves to the M3 milestone
  rather than staying in a list that no M2 commit would satisfy.

**Negative / accepted trade-offs**

- **M2 is smaller.** A reviewer looking for "did this project build a
  re-ranker" sees the answer "in M3". This is correct but slightly delays
  a headline feature. Accepted: the alternative is worse documentation.
- **A stub protocol can bit-rot.** If M3's requirements for the re-ranker
  turn out to be different from what the stub assumes, the stub will be
  rewritten. The cost is small (one file, no callers) and the alternative
  — no interface at all — means an interface designed under M3 pressure.
- **Two evaluations in M3.** M3 will publish both a retrieval-only table
  (inherited from M2) and a retrieval+re-ranker table. Slightly more work
  at M3 than if M2 had done both, but with a much stronger claim.

## References

- `docs/retrieval-and-evaluation.md` — the M2 design doc that this ADR
  scopes
- `src/recsys/retrieval/base.py` — the `IndexBackend` protocol the
  retrieval layer implements
- `src/recsys/retrieval/rerank.py` — the empty `Reranker` protocol added
  by this decision
- `README.md` — milestone table and features list, updated by this decision
- M3 milestone — where re-ranking ships and is evaluated
