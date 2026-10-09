# ADR-0015: Response cache and circuit breaker for Redis

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

Two Redis-backed behaviors are on the request path:

1. **A response cache** for retrieval results. The cache key must include
   the index version so an index swap does not serve stale neighbors
   (`docs/contracts.md` § 2.1, ADR-0007). The cache is optional: a cache
   miss is slower, not wrong.
2. **The rate limiter** (ADR-0014), which also lives in Redis and also
   has to behave when Redis is unreachable.

Both share one failure mode — Redis is down — and one design question:
what does a request do while the dependency is broken? A per-request
timeout answers "how long do we wait"; it does not answer "how many
requests do we let wait before we stop asking". Without the second
answer, a Redis outage turns into a per-request timeout on every request,
which is a slower version of an outage.

The M0 scaffold names the cache TTL (`CACHE_TTL_SECONDS`) and the
cache-hit metric (`recsys_cache_requests_total{result}`) but not the key
format, the failure behavior, or the breaker.

## Decision

Two components, one shared breaker per dependency.

### Cache

**Cache-aside.** The request path checks the cache first; on a miss it
runs retrieval and writes the result. No write-through, no read-through,
no cache invalidation on update. The reason is that the cache is keyed by
`index_version`, and an index change produces a new key space, so the old
entries expire naturally (TTL) without an invalidation message. There is
no invalidation logic to get wrong.

The cache is a **handler helper**, not a Starlette middleware. Middleware
runs before routing and cannot see the parsed request body; the cache key
includes `k`, the filter values, and the seed item ids, all of which are
part of the body. A cache middleware would have to read the body, which
means consuming and re-injecting the request stream on every request — a
cost paid by every request to serve a small fraction. The handler calls
`CacheStore.get(...)` and `CacheStore.set(...)` explicitly instead. The
trade-off is that a new handler must remember to call the store; the
counterweight is that the cache is not silently applied to endpoints it
was not designed for.

**Key:**

```
cache:v1:recommend:<blake2b-hex-16>:<variant>
cache:v1:similar:<blake2b-hex-16>
```

- `cache:v1` — a namespace prefix. `v1` is a key schema version. A
  schema change that would break existing entries increments the
  version; the old namespace is drained by TTL and then removed.
- `recommend` / `similar` — the endpoint class. Different endpoints have
  different TTLs and different cache-eligibility (below).
- `<blake2b-hex-16>` — a BLAKE2b-128 hash, hex-encoded (32 characters).
  BLAKE2b is chosen over SHA256 for the shorter output at the same
  collision resistance: 16 bytes vs 32, which halves the memory spent on
  key names for the same number of entries. The hex form is 32 characters
  instead of SHA256's 64.
- `<variant>` — the experiment variant, appended as a short suffix
  (`:control` or `:treatment`), not folded into the hash. Two variants
  of one query are two keys, and a `SCAN` for `cache:v1:recommend:*:control`
  finds every control entry. Folding the variant into the hash would make
  that impossible without a full scan.

The hash is over the canonical request: `index_version`, endpoint,
`k_max`, sorted filter fields and values, and the seed item ids **sorted
and deduplicated**. The `user_id` is **not** in the key. Personalization
at this project's scale comes from the seed items; two users with the
same seeds see the same recommendations, which is correct given the
features the model uses. Adding the user id would multiply the key space
by the user count and push the hit rate toward zero for no behavioral
change.

**`k_max`, not `k`.** The cache stores candidates at `k_max` (from
`hot_config.cache.cache_k_max`, default 100) and the handler slices to
the requested `k` at serve time. If `k` were in the key, `k=5` and
`k=20` would be two entries for one retrieval, halving the hit rate for
a request whose answer is a prefix of the other's. The cost is that a
request for `k=5` retrieves `k_max * candidate_multiplier` candidates;
the load test in M4 measures the trade-off and the config is tunable
without a restart (ADR-0022 § hot config).

**Deduplicate seeds.** `["a", "a", "b"]` and `["a", "b"]` are the
same query (a set of seeds, not a multiset); without dedup, the two
produce different keys and the second is a permanent miss.

**TTL:** per class, from config.

| Class | Default TTL | Why |
|---|---|---|
| `recommend` | 300 s (`CACHE_TTL_SECONDS`) | Short enough that a catalog change or a new index shows up quickly; long enough to absorb a repeated burst of the same query. |
| `similar` | 600 s | Item-to-item results depend only on the catalog and the index, not on a user or a session; a longer TTL is safe. |
| negative (`null` result) | 30 s | Bounded: an item that does not exist yet should start resolving as soon as it is embedded, without a full TTL wait. |

**Jitter.** The actual TTL is the configured value times a uniform random
factor in `[0.9, 1.1)`. Without jitter, a batch of entries written in the
same second expires in the same second, and the first request after
expiry pays for every entry at once (a stampede). Jitter spreads the
expiry across a window twice as wide as the TTL's precision, which is
cheap and effective.

**Negative caching.** A miss that resolves to "no such item" is stored as
a sentinel (`"\x00negative"`) for the negative TTL. A caller who probes a
nonexistent id repeatedly does not hit the retrieval path on every
request. The sentinel is distinct from a cache miss so the two do not
share a code path; a sentinel hit is a hit with a null payload, and it is
logged as a hit.

**Do not cache the re-ranked result.** The re-ranker uses recency
(ADR-0016), which is a function of wall-clock time, so the same retrieval
inputs do not produce the same output at 10:00 and 11:00. Caching the
re-ranked list would serve a stale ordering. The cache stores the
*retrieval* output (the candidate list before re-ranking); re-ranking
runs on every request. If M4's load test shows re-ranking dominates
latency, the fix is to optimize re-ranking, not to cache its output.

**MMR-enabled requests skip the cache (M3.6).** The MMR arm of the
re-ranker needs each candidate's embedding vector (ADR-0016 § MMR); a
`Candidate` with its 384-float `vector` is roughly 1.5 KB of JSON per
candidate, and caching that would bloat Redis by an order of magnitude
for a config that is not the production default. In M3.6 a request
with `enable_mmr=true` bypasses the cache lookup and the write; the
skip is counted by `recsys_cache_skip_total{reason="mmr_enabled"}` so
an operator can see the cost. The **target design** is to cache the
candidates without their vectors and, on a cache hit for an MMR
request, fetch the vectors by primary-key lookup through a future
`IndexBackend.get_vectors_by_ids(item_ids)` method. That preserves the
cache for MMR, which matters if MMR becomes the default or is A/B
tested — an A/B test of MMR against non-MMR would otherwise be
confounded by the MMR arm's 0% hit rate. The target is recorded here
so the skip is understood as a deferred feature, not a design
conclusion.

**Cache warming on index promote.** After `promote_index.py` swaps the
active pointer, the cache for the new index is empty. The first N
requests pay full retrieval. The promotion script enqueues a warmup job
that runs the top-M queries (from the golden set and the popularity
list) against the new index and writes their results with the normal
cache key. M3 supports `make index-warm` running the job manually; M6
runs it automatically after promote. The job is not on the promotion's
critical path — promotion is fast, warmup is best-effort.

**Bypass on error, never fail the request.** A cache read that raises
(connection refused, timeout, protocol error) is logged, counted as
`result="error"`, and treated as a miss. The request proceeds to
retrieval. The cache is a performance optimization, not a correctness
mechanism; a failed read must not become a failed request.

### Active index and pgvector version

Two values the request path needs are currently read from the database
on every request by `PgvectorBackend.from_registry`: the active row's
`index_version`, and the installed `pgvector` extension version (used to
decide whether `hnsw.iterative_scan` is available, ADR-0008). Both are
read once at startup and refreshed on a timer:

- **`active_index_version`** — refresh interval
  `ACTIVE_INDEX_REFRESH_SECONDS` (default 30 s). A refresh that fails
(database hiccup) keeps the previous value; it never resets to `None`.
  A `active_index.changed` log line at INFO records every transition,
  with `from` and `to`, so a metric anomaly can be correlated with an
  index promote.
- **`pgvector_version`** — read once at startup. The extension version
  does not change without a restart; a timer is not warranted.

The values live on `app.state.active_index` and
`app.state.pgvector_version`; they are passed into `sync_pipeline` as
arguments (ADR-0012 § "explicit arguments"), which is also what removes
the per-request `from_registry` call.

**Serving is gated by readiness, not by the lifespan.** If the database
is unreachable at startup, `active_index` stays `None`; the process
boots and answers `/healthz`, and `/readyz` reports `not_ready`. The
`/v1/recommend` and `/v1/items/{id}/similar` dependencies reject a
request when `active_index is None` with the same 503 envelope the
fallback-exhausted path uses, before the handler runs. The alternative
— raising in the lifespan — would put the process into a restart loop
under an orchestrator, which is a worse availability outcome than a
ready-but-not-serving pod (ADR-0019).

New metrics:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `recsys_active_index_info` | gauge | `index_version`, `model_version` | The active index the process is serving; already in `docs/contracts.md` § 4.1, this records where its values come from |
| `recsys_active_index_staleness_seconds` | gauge | — | Seconds since the last successful refresh; alert threshold is `> 2 × refresh_interval` |
| `recsys_active_index_refresh_failures_total` | counter | — | Refresh attempts that raised; a sustained rate means the database is flapping |

### Circuit breaker

**One breaker per external dependency, not per call site.** Redis has one
breaker that guards both the cache and the limiter. A breaker per call
site would let the cache breaker stay closed while the limiter breaker is
open, which is one Redis that is up for one path and down for the other —
a state that does not exist. PostgreSQL has its own breaker.

**Failure classification: connection and timeout errors only.** The
breaker counts only exceptions that mean the dependency did not answer:

- **Counted as failures:** `ConnectionError`, `TimeoutError`,
  `asyncio.TimeoutError`, `redis.ConnectionError`, `redis.TimeoutError`,
  `psycopg.OperationalError`, and connection-pool exhaustion errors.
- **Not counted as failures:** `ValueError`, `TypeError`, `KeyError`,
  `redis.ResponseError`, `redis.DataError`, `psycopg.ProgrammingError`,
  and any exception raised by a Redis or PostgreSQL response that arrives
  intact.

The rule exists because the cache and the limiter share one Redis breaker.
If a bug in the limiter raised `ValueError`, that bug would open the
breaker and take the cache down with it — a Redis outage that never
happened. A failure classification that keys on "the dependency did not
answer" keeps the breaker's state tied to the dependency's availability
and not to the caller's correctness.

**Per-instance state, not shared via Redis.** The breaker's counters
(`failure_count`, `state`, `last_failure_at`) live in the process. A
shared breaker would have to consult Redis to decide whether Redis is
down, which is the failure it exists to detect. Per-instance means that
with N replicas, N × `failure_threshold` requests must fail before every
replica opens its own breaker; the window is wider than for a single
instance but each replica still stops paying the timeout on its own. The
alternative — coordinating via a channel that is itself the dependency —
is worse.

**State machine:**

```
CLOSED  --[failure_threshold consecutive failures]-->  OPEN
OPEN    --[open_duration elapsed]------------------->  HALF_OPEN
HALF_OPEN --[1 success]---------------------------->  CLOSED
HALF_OPEN --[1 failure]---------------------------->  OPEN (reset timer, longer backoff)
```

**Consecutive failures, not a ratio.** A ratio needs a window and a
minimum sample size, both of which are parameters that have no natural
value. Consecutive failures has one parameter (`failure_threshold`) and
one behavior: N failures in a row means the dependency is not answering.
A transient failure in a healthy stream of successes does not open the
breaker.

**Failure is an exception or a timeout.** A call that raises is a
failure; a call that exceeds `timeout_seconds` is a failure. Both mean
the dependency did not answer. A call that returns a value is a success,
even if the value is "not found".

**A 4xx from the dependency is not a failure.** "Not found" from Redis
(a cache miss) and "no such index" from PostgreSQL are answers, not
errors. Counting them as failures would open the breaker on a stream of
legitimate queries for missing entries. The breaker is about whether the
dependency is available, not whether it liked the query.

**Single probe in HALF_OPEN.** The first request after the timer allows
one call through. If it succeeds, the breaker closes and normal traffic
resumes. If it fails, the breaker re-opens and the next probe is delayed.
Letting N probes through would send N requests at a dependency that just
came back, which is the load spike the breaker exists to avoid.

**Exponential backoff for the OPEN duration.** The first OPEN lasts
`open_duration_seconds` (default 5). Each subsequent trip without an
intervening success doubles the duration up to `open_duration_max_seconds`
(default 60). The doubling is capped so a dependency that is down for an
hour is probed once a minute, not once a day. A success resets the
backoff to the base value.

**Config per breaker:**

| Variable | Meaning | Default |
|---|---|---|
| `REDIS_BREAKER_FAILURE_THRESHOLD` | Consecutive failures to open | 5 |
| `REDIS_BREAKER_OPEN_SECONDS` | Initial OPEN duration | 5 |
| `REDIS_BREAKER_OPEN_MAX_SECONDS` | Cap on the backoff | 60 |
| `REDIS_BREAKER_TIMEOUT_SECONDS` | Per-call timeout | 0.1 |
| `DB_BREAKER_FAILURE_THRESHOLD` | Same, for PostgreSQL | 5 |
| `DB_BREAKER_OPEN_SECONDS` | Same | 2 |
| `DB_BREAKER_OPEN_MAX_SECONDS` | Same | 30 |
| `DB_BREAKER_TIMEOUT_SECONDS` | Same | 2.0 |

PostgreSQL's timeouts are larger because a database call that exceeds
2 seconds is a real signal, while a Redis call that exceeds 100 ms
already means the cache is not helping.

**Where the breaker is consulted.** The breaker wraps the call, not the
caller's behavior. The cache path checks the breaker before issuing the
Redis call; an open breaker is an immediate "miss, no round trip". The
limiter path checks the breaker and, if open, uses the per-instance
fallback from ADR-0014. The breaker does not decide what to do on
failure; it decides whether to try.

### Observability

New metrics:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `recsys_circuit_breaker_state` | gauge | `name` | 0 = closed, 1 = half-open, 2 = open |
| `recsys_circuit_breaker_state_changes_total` | counter | `name`, `from`, `to` | State transitions, for alerting on flapping |
| `recsys_circuit_breaker_trips_total` | counter | `name`, `reason` | `reason` ∈ `threshold` / `probe_failed` — the transition that opened the breaker |
| `recsys_cache_requests_total` | counter | `result` | Already in `docs/contracts.md` § 4.1; `result` ∈ `hit` / `miss` / `bypass` / `error` |
| `recsys_cache_negative_hits_total` | counter | — | Hits on a negative entry, useful for confirming the negative cache is earning its keep |
| `recsys_cache_write_errors_total` | counter | `type` | A read that succeeds and a write that fails is a real gap; this counts it |
| `recsys_cache_skip_total` | counter | `reason` | A request that bypassed the cache for a reason other than the breaker; `reason` ∈ `mmr_enabled` (M3.6), future reasons as they are added |

`recsys_cache_requests_total{result="bypass"}` is the metric a reviewer
should watch during an incident: a rising bypass rate means Redis is
being skipped, either because the breaker is open or because the call is
failing. It should be near zero in normal operation.

**Cardinality budget.** The label values in this section are the entire
set the project accepts:

| Metric | Allowed labels | Max cardinality |
|---|---|---|
| `recsys_circuit_breaker_state` | `name` ∈ {`redis`, `db`} | 2 |
| `recsys_circuit_breaker_state_changes_total` | `name` ∈ {`redis`, `db`}, `from`/`to` ∈ {`closed`, `half_open`, `open`} | 2 × 9 = 18 |
| `recsys_circuit_breaker_trips_total` | `name` ∈ {`redis`, `db`}, `reason` ∈ {`threshold`, `probe_failed`} | 4 |
| `recsys_cache_requests_total` | `result` ∈ {`hit`, `miss`, `bypass`, `error`} | 4 |
| `recsys_cache_negative_hits_total` | (none) | 1 |
| `recsys_cache_write_errors_total` | `type` (Python exception class name, bounded to top 5) | ≤ 5 |
| `recsys_cache_skip_total` | `reason` ∈ {`mmr_enabled`} (a closed set; a new reason is a code change) | ≤ 3 |
| `recsys_active_index_staleness_seconds` | (none) | 1 |
| `recsys_active_index_refresh_failures_total` | (none) | 1 |

`endpoint`, `user_id`, `item_id`, `request_id`, and any per-request value
are **forbidden** as labels. A label that varies per request multiplies
cardinality by the request rate; Prometheus dies of cardinality long
before it dies of sample volume. Per-request identity belongs in logs,
traces, or exemplars, not in a metric label.

Log lines:

- `cache.hit` at DEBUG.
- `cache.miss` at DEBUG.
- `cache.bypass` at WARNING when the breaker is open, with `name` and
  `reason`.
- `cache.error` at WARNING with the exception type.
- `circuit_breaker.transition` at INFO, with `name`, `from`, `to`,
  `reason`.

### No `SCAN`-based invalidation in the request path

The cache is invalidated by TTL and by the namespace version, not by an
explicit purge. `SCAN` on a live Redis is O(N) over the key space and
blocks nothing but costs CPU on the server. The namespace prefix exists
so that an operator can run a `SCAN` + `DEL` during an incident (a new
index, a corrupted value); it is not part of the steady-state path.

## Alternatives considered

| Option | Why not |
|---|---|
| **SHA256 for the cache key** | 64 hex characters vs BLAKE2b-128's 32, for the same collision resistance at this key count. The key name is stored per entry; at 1M entries the difference is 32 MB of Redis memory spent on names. BLAKE2b is faster and shorter. |
| **`xxhash` for the cache key** | Faster still, but not cryptographic. Collision resistance matters because the key derives from request content; a crafted collision would let one caller read another's cached result. BLAKE2b is the right trade-off. |
| **Fold the variant into the hash** | Makes per-variant `SCAN` impossible and hides the variant from anyone reading a key name. A separate suffix is more transparent and costs nothing. |
| **Include `user_id` in the key** | Multiplies the key space by the user count and drives the hit rate toward zero for no behavioral change: the model is not personalized by `user_id` at M3. |
| **No jitter on TTL** | Every entry written in the same second expires in the same second, producing a periodic thundering herd on the retrieval path. |
| **No negative cache** | A repeated probe for a nonexistent id hits the retrieval path every time. The negative entry costs one Redis slot for 30 seconds. |
| **Cache the re-ranked list** | The re-ranker's recency is a function of wall-clock time. Caching its output serves a stale ordering. The cache stores retrieval; re-ranking runs per request. |
| **Fail the request when the cache read fails** | Turns a cache outage into an API outage. The cache is an optimization; a missed read is a slower request, not a failed one. |
| **Breaker with a failure ratio over a window** | Two parameters with no natural value (window length, minimum sample size) and a behavior that is harder to reason about than "N in a row". Consecutive failures is one parameter and one rule. |
| **Breaker that counts 4xx as failures** | A stream of legitimate "not found" responses opens the breaker on a healthy dependency. The breaker is about availability, not about query validity. |
| **Multiple probes in HALF_OPEN** | Sends the load spike at a dependency that just came back. One probe, then close on success, is the whole point. |
| **Constant OPEN duration (no backoff)** | A dependency that is down for an hour is probed every 5 seconds, which is 720 probes for no useful signal. The backoff caps the probe rate at a defensible number. |
| **One breaker per call site** | Redis is one dependency; two breakers for one dependency is a state that cannot exist (up for one path, down for another). One breaker per dependency matches reality. |
| **Cache warming on the promotion's critical path** | Promotion should be fast and reversible; making it wait for a warmup job turns a fast operation into a slow one and gives the job veto power over the swap. Warmup runs after, best-effort. |
| **`SCAN`-based invalidation on every write** | O(N) work on a hot path for a namespace that expires on its own via TTL. The prefix is for incidents, not for steady state. |

## Consequences

**Positive**

- **A Redis outage degrades performance, not availability.** The cache
  bypasses, the limiter falls back per ADR-0014, and requests continue.
- **The breaker prevents timeout amplification.** Without it, every
  request pays the Redis timeout while Redis is down. With it, one
  request pays the failure count, then the breaker opens and the rest
  skip Redis entirely.
- **Jitter and negative caching reduce the two failure modes a naive
  cache has.** A stampede after expiry and a repeated probe for a
  missing entry.
- **The key schema is transparent.** Namespace, class, hash, variant —
  each segment has a reason, and an operator reading a key name can tell
  what it is.
- **The breaker state is a first-class metric.** An operator can see
  "Redis breaker open" in Grafana without grepping logs.

**Negative / accepted trade-offs**

- **Cache and limiter share a breaker.** A Redis failure opens the
  breaker for both, even if only one path would have failed. This is
  correct (it is one dependency) but it means the cache and the limiter
  do not degrade independently. The alternative — two breakers for one
  dependency — is a state that cannot happen.
- **Per-class TTLs are a config surface.** Three values and a jitter
  factor, each with a "why". The alternative — one TTL for everything —
  either stales the item-to-item results or serves a stale recommendation
  longer than the index's update cadence.
- **The variant suffix on keys makes the key name slightly longer.**
  Roughly 10 bytes per entry. The alternative — folding it in — loses
  the ability to `SCAN` per variant.
- **Cache warming runs the golden set and a popularity list against the
  new index.** That work is real load on the newly promoted index, done
  immediately after a swap. Mitigation: the warmup job is off the
  promotion path and can be run manually if the index is known to be
  fragile; a future ADR adds rate limiting to the warmup job if it
  becomes a problem.
- **The breaker's per-dependency timeouts are different numbers.** Redis
  at 100 ms and PostgreSQL at 2 s is a judgment, not a derivation. A
  reviewer who disagrees can change the config; the defaults are
  recorded with their reasoning.
- **Two Redis clients, one breaker.** The limiter and the cache each
  hold their own `redis.asyncio.Redis` connection. They share the breaker
  (correctly — one dependency, one breaker) but not the connection pool.
  The two clients are built through a single `make_redis_client(cfg)`
  factory so their `socket_timeout`, `max_connections`, and
  `decode_responses` cannot drift. A future change consolidates to one
  shared client on `app.state.redis`; the refactor touches the app
  factory, `deps.py`, and every test that constructs a client, so it is
  deferred. The cost today is one extra pool of at most ten connections.
- **No `SCAN`-based purge is exposed as a CLI in M3.** The namespace
  prefix exists for the operator to use manually. A `make cache-purge`
  target is a future addition; M3 documents the `SCAN` pattern in
  `docs/ops.md` and leaves the command as the operator's shell.

## References

- `docs/adr/0007-index-version-identity.md` — the `index_version` the
  cache key includes
- `docs/adr/0014-rate-limiting.md` — the second Redis-backed path this
  breaker guards
- `docs/contracts.md` § 2.1 — the response shape the cache stores
- `docs/contracts.md` § 4.1 — the cache metric
- `docs/ops.md` — the `SCAN` purge pattern and the cache-warming runbook
- `src/recsys/cache/store.py` — the cache-aside store the handler calls
  after the request body is parsed. It is **not** a Starlette middleware:
  the cache key includes `k`, filters, and seeds, so the body must already
  be parsed when the lookup runs. Starlette middleware exists for
  cross-cutting concerns (auth, rate limit, request context); the cache is
  not one.
- `src/recsys/resilience/breaker.py` — the breaker implementation. It
  lives under `resilience/` (with retry, bulkhead, timeout as future
  siblings), not under `monitoring/`. `monitoring/` is for observing the
  system (metrics, logs, traces, health); the breaker changes what the
  system does under failure. It emits a metric, but a component is not
  categorized by one of its outputs.
- `tests/integration/test_cache.py` and `tests/integration/test_breaker.py`
