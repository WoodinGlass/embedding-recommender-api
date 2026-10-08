# ADR-0020: Fallback chain

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

The retrieval path has three things that can fail independently:

- **Redis** — the cache and the rate limiter. Optional by design
  (ADR-0015).
- **The ANN index** — the pgvector HNSW index. When it is slow or
  unavailable, the retrieval path cannot answer from it.
- **PostgreSQL** — the store that backs the index. When it is down, the
  ANN path is down with it.

The M0 scaffold names a three-tier chain in `README.md` § Retrieval
("cache", "ann", "fallback") and `docs/contracts.md` § 2.1 (the
`meta.source` enum). It does not distinguish between the two ways the
fallback can be reached (an ANN timeout versus a database outage), and
it does not say what happens when the fallback itself is not reachable.

The review of the first draft rejected "503 when both ANN and the
fallback fail" as too narrow. A database outage that stops the
recommendation path entirely is a worse outcome than a recommendation
path that serves a pre-computed popular list from memory, and the
project has the data to serve that list without touching the database.

## Decision

### The chain has four tiers, and `503` is the fifth

| Tier | `meta.source` | Served when |
|---|---|---|
| 1 | `cache` | A cache hit for the exact request (ADR-0015). |
| 2 | `ann` | The filtered ANN query on the active index returns within `ANN_TIMEOUT_MS` and the re-ranker (ADR-0016) produces a list. |
| 3 | `fallback_ann` | The ANN path failed or timed out, but the database is reachable. A pre-computed popular list, read from the database, is filtered by the request's filters and returned. |
| 4 | `fallback_cached` | The database is unreachable (its circuit breaker is open or the query raises), so tier 3 cannot run. The same popular list, cached in process memory, is returned, with the request's filters applied in memory. |
| 5 | *(503, `meta.source: "none"`)* | None of the above produced a non-empty list. |

The chain tries tiers in order and stops at the first that produces a
non-empty list. A tier that produces an empty list (e.g. the request's
filters match no item in the cached popular list) falls through to the
next tier, because "no items" is not an answer.

### Tier 3 — `fallback_ann` reads from the database

The "popularity" list for tier 3 is a small table
(`popularity_snapshot`) holding a pre-computed ranking, refreshed by a
background job (in M3, a `make popularity-refresh` target that an
operator or a cron runs; in M6, the CD pipeline runs it after every
promote). The table is small (top 1 000 items, or the top N where N is
config, `POPULARITY_SNAPSHOT_SIZE`, default 1 000) and has the item's
`category`, `brand`, and `language`, so the request's filters can be
applied in SQL.

The query for tier 3 is:

```sql
SELECT item_id, rank
FROM popularity_snapshot
WHERE (category = %(category)s OR %(category)s IS NULL)
  AND (brand    = %(brand)s    OR %(brand)s IS NULL)
  AND (language = %(language)s OR %(language)s IS NULL)
ORDER BY rank ASC
LIMIT %(k)s
```

The filter fields come from the `FILTER_FIELDS` allowlist (ADR-0008);
the query is parameterized; no field name is ever built from user input.

The `rank` is a small integer, and the `score` the response carries is
`1.0 - rank / (snapshot_size + 1)`, so scores are in `(0, 1]` and the
first item's score is `1.0`. The score is a placeholder that satisfies
the schema's `[0.0, 1.0]` constraint; clients that care about similarity
should look at `meta.source` and know that a fallback score is not
comparable to an ANN score.

### Tier 4 — `fallback_cached` reads from process memory

The same list is loaded into memory at startup and refreshed every
`POPULARITY_CACHE_REFRESH_SECONDS` (config, default 300). The in-memory
shape is a small list of `(item_id, rank, category, brand, language)`
tuples; a filter is a comprehension over it. At N = 1 000, a filter is
microseconds.

The in-memory list is refreshed by a background task started in the
application's lifespan. If the refresh fails (the database is down),
the previous list stays in memory and a `popularity.cache.stale` log
line records the failure. The list is never empty after the first
successful load; a process that starts with the database down has an
empty list and tier 4 falls through to tier 5. The startup log prints
the row count so an operator sees the state.

**Why memory, not Redis.** Redis is a dependency that can be down for
the same reason PostgreSQL is (a network partition, a pod restart). A
fallback that lives in Redis is a fallback that fails with the same
class of failure as the primary. Process memory fails only when the
process fails, which is when the instance is already out of rotation
(ADR-0019).

### ANN timeout

`ANN_TIMEOUT_MS` (config, default 120 ms; `docs/contracts.md` § 3) bounds
the ANN query. A query that exceeds it is cancelled and the chain moves
to tier 3. The timeout is configurable because the right value depends
on the deployment's latency budget, which M4 measures.

The re-ranker runs inside the ANN tier's budget: the ANN query and the
re-rank together must fit within the latency target. If the ANN query
returns in 80 ms and the re-ranker takes 60 ms, the tier's total is
140 ms, which exceeds the target; the chain does not enforce a joint
budget in M3, but M4's load test measures whether one is needed.

### Circuit breakers gate tiers 3 and 4

Tier 3 consults the PostgreSQL breaker (ADR-0015). An open breaker is an
immediate fall-through to tier 4 with no query attempted. This is the
mechanism that makes the "database is down" case cheap: the first
request after the outage pays the failure count, then the breaker
opens, and the rest skip the query.

Tier 4 consults no breaker; it reads memory.

Redis's breaker gates the cache tier: an open Redis breaker makes tier 1
a miss, and the chain proceeds to tier 2.

### `meta.source` reflects the tier that produced the list

The response's `meta.source` is one of `cache`, `ann`, `fallback_ann`,
`fallback_cached`, or `none` (the last only in the 503 response). The
enum in `docs/contracts.md` § 2.1 is extended from three values to five;
the change is noted in this ADR.

**Why the two fallback tiers have different names.** An operator
watching the fallback rate needs to know which tier the traffic is on.
`fallback_ann` means the ANN path is unhealthy; `fallback_cached` means
the database is unhealthy. The remediations are different (Runbook 1
for the database, Runbook 2 for latency), and a single `fallback` value
would collapse two signals into one.

### What "fall through" means for the response

A request that lands on tier 3 or 4 returns a valid response with a
populated `items` list and a `meta` object whose `source` names the
tier. The response's status code is 200, not 503: the client asked for
a recommendation and got one. The quality is lower; the availability
is not.

A request that lands on tier 5 returns 503 with a body whose
`meta.source` is `none`. The `error.code` is `unavailable`; the
`error.details.reason` names why (`no_tier_produced_results`).

### Guardrail and alert

The fallback rate — the fraction of requests served from tiers 3 or 4
— is a guardrail metric (ADR-0010 § Guardrails). The target is below 1%
of requests; above that for five minutes is an alert (the thresholds and
the alert live in the M4 dashboards, but the metric is emitted in M3).

The metric is `recsys_fallback_total{tier}` where `tier` is
`fallback_ann` or `fallback_cached`. The M0 metric was
`recsys_fallback_total{reason}`; the label is renamed in M3 to
`tier` and the values are the tier names. The existing tests and
dashboards that reference `reason` are updated in the same PR; the
metric is not yet in production.

### Log lines

Every fallback transition logs one line:

- `retrieval.fallback.ann` at WARNING, with `reason` (`timeout` /
  `error`), `ann_timeout_ms`, and `latency_ms` of the failed attempt.
- `retrieval.fallback.cached` at WARNING, with `reason`
  (`db_breaker_open` / `db_error`), `cache_age_seconds`, and
  `row_count`.
- `retrieval.fallback.none` at ERROR, with the reasons every tier failed.

A tier-3 or tier-4 response is a signal, not an error. Its log line is
at WARNING because a reviewer should see it, not at ERROR because it is
not a bug in this request.

### What is not in the chain

- **No "retry the ANN query once before falling through".** A query
  that exceeded its timeout once will exceed it again under the same
  load. The retry doubles the latency budget for the failing request and
  adds the same load to the dependency that just failed. The chain moves
  on.
- **No per-request budget across tiers.** Each tier's cost is bounded
  (ANN by `ANN_TIMEOUT_MS`, tier 3 by the query's own timeout, tier 4 by
  a comprehension in memory). A joint budget (total 200 ms across all
  tiers) is a possible refinement; M3 keeps the tiers independent, and
  M4 measures whether the joint budget is needed.
- **No partial cache.** A cache hit is the whole list; there is no
  "merge cache and ANN". A partial merge would produce a response whose
  ordering no client can reason about (half cached, half fresh) and
  whose `meta.source` would have to be a new value. Not worth the
  ambiguity.

### What this ADR does not cover

The re-ranker's own failure (a signal provider that raises) is handled
inside the re-ranker (ADR-0016 § Failure behavior): the re-ranker
returns the retrieval list and the response is `meta.source = "ann"`.
This ADR is about the tiers above retrieval, not about failures within
the ANN tier.

## Alternatives considered

| Option | Why not |
|---|---|
| **503 when ANN and the fallback both fail (three tiers)** | The first draft. A database outage stops recommendations even though the project has a pre-computed popular list in memory. The cost of the fourth tier is small (a background task and an in-memory list); the benefit is that a database outage is a quality degradation, not an outage. |
| **Fallback cached in Redis** | Redis is a dependency that can fail with the same network partition that took PostgreSQL. A fallback in process memory fails only with the process, which `/livez` already catches. |
| **Fallback recomputed from the last successful ANN result** | The last successful result is per request, not a global list; there is no "last successful" for a request the instance has never seen. A pre-computed global list is what the fallback is. |
| **Fallback reads `item` table live (no snapshot)** | A live read of the whole catalog to rank by a popularity field is not a fallback; it is a second query path that has to be tuned and can fail. A pre-computed snapshot is one indexed lookup. |
| **One `fallback` value for both tiers** | Collapses two different operational signals (the ANN is slow vs the database is down) into one metric and one log value. The remediations are different; the values are different. |
| **`meta.source` extended with a `degraded` value** | Would conflate "how the list was produced" with "the system's state". The `meta.source` field names the source; the readiness endpoint (ADR-0019) names the state. |
| **Retry the ANN query once before falling through** | Doubles the failing request's latency and re-loads the dependency that just failed. A retry with backoff is a caller's tool; the server's job is to answer within its budget. |
| **Per-request budget across tiers** | A refinement that M3 does not need. If M4 measures that a joint budget would improve p95, an ADR adds it. Keeping the tiers independent in M3 makes each tier's behavior measurable in isolation. |
| **Empty list from tier 3 falls through to 503** | An empty list is not an answer for a request that had a filter; falling through to tier 4 (which may have a wider list in memory) is the right behavior. Tier 4 with the same filter producing an empty list falls through to 503. |
| **Tier 3 without a circuit breaker** | Every request during a database outage pays the query timeout. The breaker is what makes the outage cheap; ADR-0015 defines it for PostgreSQL and this tier consults it. |
| **`meta.source` kept as three values, with `fallback` covering both tiers** | Same problem as a single fallback value: an operator cannot distinguish two remediations. Extending the enum is a contract change; the ADR records it. |
| **Log the tier-3/tier-4 response at INFO** | A fallback is a signal an operator should notice. WARNING gets it into the log stream a reviewer watches; INFO would bury it among normal request lines. |
| **Alert only on `fallback_cached`, not on `fallback_ann`** | Both are degradations with a defined threshold. A sustained `fallback_ann` rate is the early warning of a latency problem that would become `fallback_cached` if the database followed. |

## Consequences

**Positive**

- **A database outage does not stop the recommendation path.** The
  in-memory popular list serves a lower-quality response at full
  availability. `503` becomes the case where the process has never seen
  a popular list, which is a first-request-after-cold-start scenario,
  not an outage scenario.
- **The two fallback tiers have different `meta.source` values and
  different metrics.** An operator can see which dependency failed from
  the metric alone.
- **Tier 3's query is bounded and parameterized.** A single indexed
  lookup on a small table, with the same filter allowlist the primary
  path uses.
- **Tier 4's cost is a comprehension over 1 000 tuples.** Microseconds;
  no dependency to fail.
- **The chain's cost is bounded per tier.** ANN by its timeout, tier 3
  by the query, tier 4 by memory. A request cannot spend more than the
  sum of the tiers' bounds, and each bound is configurable.

**Negative / accepted trade-offs**

- **The `meta.source` enum grows from three values to five.** A client
  that switches on the field exhaustively has to handle the new values.
  The change is documented in `docs/contracts.md` § 2.1 and in the
  ADR's Change management section; the API is not yet in production, so
  there is no client to migrate.
- **The fallback score is not a similarity.** A client that compares a
  fallback score to an ANN score would be comparing two different
  quantities. The `meta.source` field is the signal; the score's
  meaning is documented per source.
- **Tier 3 requires the `popularity_snapshot` table and its refresh.**
  A background task, a config value for the snapshot size, and a
  `make popularity-refresh` target. Small, but not free.
- **Tier 4's in-memory list is a second copy of the same data.** At 1 000
  rows it is a few tens of kilobytes; the duplication is the price of
  not depending on Redis.
- **A process that starts during a database outage has no tier 4 list.**
  The first requests fall through to `503` until the database comes back
  and the background refresh succeeds. The startup log records the
  empty state so an operator sees it.
- **The fallback rate target (1%) is a guardrail, not a gate.** A
  sustained rate above it alerts but does not fail a build. The
  decision to make it a gate is M5's (experiment guardrails); M3 emits
  the metric and the alert, and leaves the policy to the experiment
  layer.
- **`fallback_ann` and `fallback_cached` are two new enum values in
  `docs/contracts.md` § 2.1.** The section is updated in the same PR;
  the change is additive.

## References

- `docs/adr/0008-filter-strategy.md` — the `FILTER_FIELDS` allowlist
  tier 3's query uses
- `docs/adr/0010-evaluation-thresholds.md` § Guardrails — the
  fallback-rate guardrail
- `docs/adr/0014-rate-limiting.md` — the fallback pattern (per-instance)
  this ADR echoes for retrieval
- `docs/adr/0015-cache-and-circuit-breaker.md` — the PostgreSQL and
  Redis breakers tiers 3 and 4 consult
- `docs/adr/0016-reranker-composition.md` § Failure behavior — the
  re-ranker's own fallback (inside tier 2, not in this chain)
- `docs/adr/0019-readiness-contract.md` — the state endpoint that
  reports the dependencies' health
- `docs/contracts.md` § 2.1 — the `meta.source` enum this ADR extends
- `src/recsys/fallback/popularity.py` — the in-memory list and its
  refresh
- `src/recsys/api/routers/recommend.py` — the chain
- `tests/unit/test_fallback_chain.py` and
  `tests/integration/test_fallback_paths.py`
