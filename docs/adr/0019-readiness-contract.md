# ADR-0019: Readiness and liveness contract

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

The M0 scaffold exposes `/healthz` (liveness) and `/readyz` (readiness),
and the M0 `create_app` returns `{"status": "ok"}` and `{"status":
"ready", "checks": {}}` respectively. `docs/contracts.md` § 2 promises
that readiness means "database reachable and an active index loaded"
and that Redis status is "reported, not required". None of that is
implemented yet.

M3 makes the retrieval path live behind the API, which means three
dependencies can fail in ways the probe has to distinguish:

- **PostgreSQL** — the retrieval path cannot answer without it. Its
  failure is a real outage.
- **The active index** — the API can talk to PostgreSQL and still have
  no index to query against (a fresh database, a failed promotion, a
  replica that has not caught up). Its absence is an outage too.
- **Redis** — the cache and the rate limiter are optimizations
  (ADR-0014, ADR-0015). A Redis outage degrades performance, not
  correctness. It is not a reason to drain an instance.

Three more questions:

1. **What does a probe cost?** A Kubernetes readiness probe runs every
   few seconds on every pod. At ten pods and a five-second interval that
   is two requests per second per check, all of which hit the same
   dependencies. If every probe re-queries the database, the health
   check becomes a load source.
2. **What does the status field mean?** A boolean plus a "not required"
   nullable third is ambiguous (see the rejected alternative below).
3. **What happens when the probe itself times out?** A check that hangs
   is worse than a check that fails: an orchestrator that cannot read
   the probe response will retry, and the retries pile up on a
   dependency that is already slow.

## Decision

### Two endpoints, two questions

| Endpoint | Question | Kubernetes use |
|---|---|---|
| `/livez` | Is this process alive enough to keep running? | `livenessProbe` |
| `/readyz` | Should this instance receive traffic right now? | `readinessProbe` |

`/healthz` is kept as an alias of `/livez` for backwards compatibility
with the M0 scaffold and with any external caller that adopted it. It
returns the same body as `/livez`.

### `/livez` — liveness only

`GET /livez` returns `200` with:

```json
{ "status": "alive" }
```

The handler touches no dependency: no database, no Redis, no filesystem
beyond what the response itself needs. Its purpose is to detect a
process that is dead (a deadlock, a corrupted event loop, an OOM that
has not yet killed the pod). A liveness probe that checked the database
would restart a healthy pod when the database is briefly unreachable,
which turns a database hiccup into a restart storm.

### `/readyz` — readiness with three states

`GET /readyz` returns:

```json
{
  "status": "ready",
  "checks": {
    "db":    { "ok": true,  "latency_ms": 2 },
    "index": { "ok": true,  "index_version": "idx-a3f9e021", "row_count": 200 },
    "redis": { "ok": true,  "required": false, "latency_ms": 1 }
  }
}
```

**`status` is one of three values**, derived from `checks`:

| Value | Meaning | HTTP |
|---|---|---|
| `ready` | Every required check is `ok`. | 200 |
| `degraded` | Every required check is `ok`; at least one optional check is not. The instance can serve, more slowly or with less functionality. | 200 |
| `not_ready` | At least one required check is not `ok`. The instance should be removed from rotation. | 503 |

`db` and `index` are required. `redis` is optional (`required: false`
in its body). The rule is stated in the response so a caller does not
have to read this ADR to know which check can be ignored.

**HTTP status** is 200 for both `ready` and `degraded`. Returning 503
for `degraded` would drain a healthy instance on a Redis hiccup, which
is exactly the failure mode the `degraded` state exists to prevent. The
body is where the difference lives; the status code is "can this
instance serve" (yes for both).

### What each check does

**`db`:** runs `SELECT 1` on a connection acquired from the pool, with
a bounded timeout. Measures the time from acquire to result. Reports
`ok` on success, the exception's category on failure. A failed `db`
means the instance cannot answer retrieval queries.

**`index`:** queries `index_registry` for the row with `status = 'active'`
and reads its `index_version` and `row_count`. `ok` is true when exactly
one row is active (the partial unique index guarantees at most one; the
check verifies at least one) and the corresponding `embedding` table
has at least one row. Reports the `index_version` in the body, so a
caller can see which index is being served.

**`redis`:** runs `PING` with a bounded timeout. `ok` on `PONG`. The
check is marked `required: false` in the body. A failed Redis does not
change `status` to `not_ready`; it changes it to `degraded` if every
required check is `ok`.

### Result caching: 5 seconds, per process

The full set of checks runs at most once every `READYZ_CACHE_SECONDS`
(config, default 5). Within the cache window, a second call returns the
cached result without touching any dependency.

**Why a cache is necessary.** Kubernetes readiness probes run every few
seconds on every pod. Without a cache, N pods × probes-per-minute hits
the database with N × 12 × 60 queries per hour, all of which are
identical. A 5-second cache reduces that by the ratio of the probe
interval to the cache window, which for a 5-second probe is one (no
reduction) and for a 10-second probe is two (half the load). The exact
factor depends on the deployment's probe interval; the point is that
the cache bounds the cost of the probe to "at most once every five
seconds per process", which is a number an operator can reason about.

**Why 5 seconds and not 60.** A cache that outlives the deployment's
probe interval adds latency to the drain signal: when the database
comes back, the instance should be ready within seconds, not minutes.
A cache that is shorter than the probe interval is wasted work. Five
seconds is short enough for a fast recovery and long enough that a
5-second probe hits the cache on the second request.

**Cache scope is per process, not shared.** A shared cache (in Redis)
would defeat the purpose: the readiness check would depend on the
dependency it is trying to observe. Per-process caching means each
instance's probe reflects that instance's own view, which is what a
readiness probe is for.

### Per-check timeout

Each check has a hard timeout:

| Check | Timeout | Config |
|---|---|---|
| `db` | 500 ms | `READYZ_DB_TIMEOUT_SECONDS` |
| `index` | 500 ms | `READYZ_INDEX_TIMEOUT_SECONDS` |
| `redis` | 100 ms | `READYZ_REDIS_TIMEOUT_SECONDS` |

A check that exceeds its timeout is treated as failed and its entry
carries `error: "timeout"`. The overall handler has a ceiling as well:
if all three checks are slow, the handler still returns within ~1.1
seconds, because they run with their own timeouts.

**A timeout is a failure, not a warning.** The purpose of a readiness
probe is to answer quickly. A database that takes 400 ms to answer a
`SELECT 1` is a database the retrieval path cannot serve from within
its latency budget; reporting `ok` with a high `latency_ms` would
keep traffic on an instance that cannot serve it. The `latency_ms` is
reported anyway, so an operator watching the endpoint sees the trend
before it becomes a failure.

### No caching of failure into success

The cache stores the check result verbatim. A failed check is cached
as failed for the cache window; the next probe within the window
returns the failure. The alternative — retrying a failed check on every
probe — re-queries a dependency that is already known to be down, which
is the load amplification the cache exists to avoid.

### Which paths bypass the probe cache

None. `/livez` does not use the cache (it does not check anything).
`/readyz` always uses the cache, including the first call in a process
(which populates it).

### What is not checked

- **Disk space.** Useful, but a filesystem check that fills the probe
  body with a percentage is not what a readiness probe is for. The
  alert lives in Prometheus (`node_filesystem_avail_bytes`), not in
  `/readyz`.
- **Upstream identity provider (JWT)** — M3 uses HS256 with a shared
  secret, so there is no upstream. If a future ADR adds RS256 with a
  JWKS endpoint, the JWKS fetch is a dependency that readiness could
  check; not now.
- **Model artifact availability.** The ONNX artifact is loaded at
  startup and held in memory; it cannot become unavailable without the
  process crashing, which `/livez` catches.
- **Event-store health.** The event store is the same PostgreSQL the
  `db` check covers; a separate check would double the probe's cost for
  the same signal.

## Alternatives considered

| Option | Why not |
|---|---|
| **`redis: null` for a down Redis** | Ambiguous: does null mean "not configured", "not checked", or "down"? A client cannot act on it. The structured `{ok, required, error}` form makes all three cases distinguishable. |
| **`status` as a boolean plus a nullable Redis** | The consumer has to translate two fields into three states. The explicit three-valued `status` puts the interpretation in the response. |
| **503 for `degraded`** | Drains a healthy instance on a Redis hiccup. The instance can serve; the response says so; an orchestrator that removed it would be removing capacity the system needs. |
| **No cache on `/readyz`** | Turns the readiness probe into a load source. At ten pods and a 5-second probe interval that is two database round trips per second per check, from the orchestrator, forever. |
| **Cache the readiness result in Redis** | Makes the readiness check depend on Redis to answer a question about Redis's own availability. Per-process caching is what a readiness probe is for. |
| **Longer cache (60 seconds)** | Adds up to a minute of latency to the drain signal in both directions: a healthy dependency's recovery does not mark the instance ready for up to a minute, and a broken one does not mark it not-ready. The probe interval is the natural ceiling; 5 seconds is a reasonable default. |
| **A timeout is a warning, not a failure** | A database that takes 400 ms to answer `SELECT 1` cannot serve a request whose budget is 200 ms. Reporting `ok` keeps traffic on an instance that cannot serve it. The `latency_ms` field is the trend signal; the `ok` field is the action signal. |
| **Per-check retries inside `/readyz`** | Retrying a check that is already known to be slow re-queries the dependency the check just observed to be failing. The cache is the right way to bound retries. |
| **A single `/healthz` that does both liveness and readiness** | The two questions have different failure modes. A liveness failure means "restart me"; a readiness failure means "stop sending traffic". A single endpoint that answers both has to pick one action, and the wrong pick is a restart storm (checking dependencies in liveness) or a dead pod serving traffic (checking only the process in readiness). |
| **Drop `/healthz` in M3** | The M0 scaffold and the CI Docker smoke test both call `/healthz`. Keeping it as an alias of `/livez` is one line and no ambiguity; removing it would break the CI job that M0 shipped. |
| **Include disk space in `/readyz`** | The check is a different kind (a percentage threshold, not a boolean) and its alerting belongs in the metrics stack. Adding it to `/readyz` would mix two operational questions in one response. |
| **A `checks` object with only the failing checks present** | A client reading the response would have to distinguish "check passed" from "check not run". The structured form lists every check every time, with its own `ok`. |

## Consequences

**Positive**

- **The three-state status is unambiguous.** A client (orchestrator,
  load balancer, dashboard) can act on `ready` / `degraded` /
  `not_ready` without reading the endpoint's documentation.
- **Liveness and readiness are separate.** A database hiccup does not
  restart the pod; a deadlocked process does not keep receiving
  traffic.
- **The probe cost is bounded.** At most one full check per process
  per `READYZ_CACHE_SECONDS`. An operator can compute the load the
  probes put on the database.
- **`degraded` keeps the instance in rotation.** A Redis outage reduces
  capacity but does not create an outage where there was none.
- **The response names the active index.** A caller (or a dashboard)
  can see which `index_version` an instance is serving without a second
  query.

**Negative / accepted trade-offs**

- **A failed check is cached as failed.** During the cache window, the
  probe keeps reporting the failure even if the dependency has just
  recovered. The recovery latency is bounded by the cache window (5 s
  default), which is short. A shorter window would increase the probe
  load; the trade-off is stated and configurable.
- **Three config values for timeouts.** Each has a reason (the numbers
  reflect the latency budget of the corresponding path) but they are
  three more knobs. The defaults are documented in `.env.example` and
  in `docs/contracts.md` § 3.
- **A slow-but-not-failed database marks the instance not-ready.** The
  threshold is the per-check timeout (500 ms), which is a judgment
  about what "slow enough to fail" means. An operator who disagrees
  can raise the timeout, at the cost of keeping a slow instance in
  rotation longer.
- **`/healthz` is kept indefinitely.** It is an alias with no behavior
  of its own. A future ADR can deprecate it if the alias ever becomes a
  source of confusion; removing it now would break the M0 CI job.
- **The `latency_ms` field is present but not thresholded.** A client
  cannot use it to make a decision on its own; it is a monitoring
  signal. A future ADR could add a `slow` state if the deployment needs
  one, but the three-state model is enough for M3.

## References

- `docs/contracts.md` § 2 — the endpoint list and the readiness promise
- `docs/adr/0015-cache-and-circuit-breaker.md` — the Redis
  degradation that `degraded` reports
- `docs/runbook.md` — Runbook 1, a failing `/readyz`
- `src/recsys/api/routers/health.py` — the endpoints
- `src/recsys/api/readiness.py` — the check runner and its cache
- `tests/unit/test_readiness.py` and
  `tests/integration/test_readyz.py`
