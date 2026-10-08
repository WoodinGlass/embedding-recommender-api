# ADR-0014: Rate limiting (token bucket in Redis, per-credential)

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

The API is called by a small number of credentials, each of which may
drive many requests. Without a limiter, a single misbehaving caller can
exhaust the retrieval path for everyone. The M0 scaffold names a
per-minute limit (`RATE_LIMIT_PER_MINUTE`) and Redis as the shared store
(`docs/contracts.md` § 3, § 2.5), but does not fix the algorithm, the
key, the failure mode, or the observability.

Four questions must be answered before code:

1. **What algorithm?** The obvious choice — a sliding window implemented
   as a Redis sorted set — is the wrong choice at this scale. Memory is
   O(N) per key, where N is the number of requests in the window; at
   10 000 active keys and a 1 000-request-per-minute limit, the sorted
   sets hold ten million entries. Every request is an O(log N) insert
   plus a range query, and every expired request must be reaped with
   `ZREMRANGEBYSCORE` or the memory grows without bound.
2. **What key?** The M0 scaffold does not say. Per-IP is common but
   wrong: it punishes everyone behind a NAT and misses one credential
   fanned across many source addresses.
3. **What happens when Redis is down?** A single-instance in-memory
   limiter with the full configured limit exceeds the intended total on
   a multi-instance deploy. A refusal to serve traffic at all is worse
   than the alternative.
4. **What does the client see?** `429` and a `Retry-After` header are
   named in `docs/contracts.md` § 2.5 but the value is not specified.

## Decision

### Algorithm: token bucket, executed as a Lua script in Redis

A bucket per `(credential, endpoint)` holds two values: the current token
count and the last-refill timestamp. On each request the bucket is
refilled based on the elapsed time since `last`, then one token is
deducted. If the bucket cannot pay, the request is refused.

The algorithm is chosen for its cost profile:

- **Memory is O(1) per key.** Two numbers, regardless of request rate.
- **One round trip per request.** The read-modify-write is a single Lua
  script evaluated server-side; there is no race between the read and
  the write, and there is no separate "reap" pass.
- **Bursts are absorbed up to the bucket size.** A caller who has been
  quiet for a minute does not get punished for a burst that fits inside
  the limit.
- **Smoothing is configurable.** The refill rate and the bucket size are
  two independent parameters, which is enough to express "1 000 per
  minute, bursts of 100" without inventing a new windowing scheme.

The Lua script is:

```lua
-- KEYS[1] = rate_limit:{credential_hash}:{endpoint}
-- ARGV[1] = capacity (bucket size)
-- ARGV[2] = refill_rate (tokens per second)
-- ARGV[3] = now (unix seconds, fractional)
-- ARGV[4] = cost (tokens this request consumes; normally 1)
local tokens = tonumber(redis.call('HGET', KEYS[1], 'tokens') or ARGV[1])
local last   = tonumber(redis.call('HGET', KEYS[1], 'last')   or ARGV[3])
local elapsed = tonumber(ARGV[3]) - last
if elapsed > 0 then
  tokens = math.min(tonumber(ARGV[1]), tokens + elapsed * tonumber(ARGV[2]))
end
local cost = tonumber(ARGV[4])
if tokens < cost then
  local deficit = cost - tokens
  local wait = math.ceil(deficit / tonumber(ARGV[2]))
  return {0, math.floor(tokens), wait}
end
tokens = tokens - cost
redis.call('HSET', KEYS[1], 'tokens', tokens, 'last', ARGV[3])
-- Expire well after the bucket would have refilled from empty, so an
-- idle key does not live forever.
redis.call('EXPIRE', KEYS[1], math.ceil(tonumber(ARGV[1]) / tonumber(ARGV[2])) * 2)
return {1, math.floor(tokens), 0}
```

The script returns `(allowed, remaining, retry_after_seconds)`. The
retry-after value is computed from the deficit and the refill rate, so
the `Retry-After` header is a real number derived from the bucket's
state, not a fixed "try again in a minute".

### Key: `(credential hash, endpoint)`

The key is built from:

- A fixed namespace prefix (`rate_limit:`), so a Redis `SCAN` can find
  and (if needed) invalidate the whole set.
- A BLAKE2b hash of the presented credential, truncated to 16 bytes. The
  raw credential is never stored (ADR-0013 § No credential in logs,
  metrics, or errors).
- The endpoint template, not the resolved path. `/v1/items/{item_id}/similar`
  is one key; every distinct `item_id` does not get its own bucket.
- The limit class, not the concrete limit. `recommend` and `events` have
  different limits (below); the key names the class, and the class maps to
  a limit read from config.

Per-credential, not per-IP. The credential is the identity the API
authenticates (ADR-0013); if it is trusted enough to authorize the
request, it is the right unit to meter. A single credential used from
many IPs is one caller; many callers behind one NAT are many, and the
bucket is per credential so each gets its own.

### Per-endpoint limits

Rate limits are not one number for the whole API. The classes are:

| Class | Endpoints | Rationale |
|---|---|---|
| `recommend` | `POST /v1/recommend`, `GET /v1/items/{id}/similar` | The expensive path: cache lookup, ANN query, re-rank. |
| `events` | `POST /v1/events` | Write path, but cheap per event; higher limit than `recommend`. |
| `admin` | Index promote, rollback (M6) | Rare, but a runaway loop against promote is a real incident. |
| `unlimited` | `/healthz`, `/livez`, `/readyz`, `/metrics` | Probes must never be rate limited: a `429` on readiness drains the instance. |

The limit per class lives in config
(`RATE_LIMIT_RECOMMEND_PER_MINUTE`, `RATE_LIMIT_EVENTS_PER_MINUTE`, ...).
`RATE_LIMIT_PER_MINUTE` from M0 remains as the default for classes that
do not override it.

Health probes are `unlimited`. This is not a convenience: a rate limit on
`/readyz` means a busy instance reports `429`, the orchestrator reads
that as "not ready", and drains a healthy pod. The path is also cheap
(no DB, no ANN), so the attack surface is small.

### Failure mode: per-instance fallback with a split limit

When the Lua script cannot be evaluated (Redis unreachable, breaker open,
script missing), the limiter degrades to an in-process token bucket with
the **configured limit divided by the number of running instances**.

The division is the whole point. A naive fallback that keeps the full
limit on each instance multiplies the effective limit by the instance
count: at four instances, a 1 000-per-minute limit becomes 4 000. The
division keeps the *aggregate* close to the configured limit at the cost
of being slightly strict when a caller is unlucky enough to have its
requests spread evenly across instances.

The instance count comes from a shared source (a Redis key, an
environment variable set by the deployment, or a per-instance
`INSTANCE_COUNT` config). M3 supports the environment variable, since
the project deploys as a small fixed set of containers; a service
registry is a future ADR if the fleet becomes dynamic.

The fallback logs at WARNING on first use (not per request) and sets
`recsys_rate_limit_degraded` to 1. The metric is the alerting signal;
the log line is the "why". Both clear when the primary limiter works
again.

### Failure mode: breaker

The Redis call for the limiter sits behind the circuit breaker defined
in ADR-0015 (shared with the cache). One breaker for the Redis
dependency, not one per call site. The limiter and the cache fail
independently only in the sense that a cache failure does not open a
limiter breaker; a Redis failure opens both at once because the
dependency is the same.

### What is not rate limited

- **Health checks** (`/healthz`, `/livez`, `/readyz`) and `/metrics`.
  See above: a rate-limited probe drains a healthy instance.
- **Requests that fail authentication.** The limiter runs after the auth
  dependency, so an unauthenticated request costs an auth failure, not a
  token. This also keeps an attacker from exhausting a victim's bucket
  by spamming with the victim's (unknown) credential.
- **Admin calls during an incident.** Promoting an index is the response
  to an incident, and the operator's credential should not be throttled
  by it. The `admin` class has a high limit and is not the path that a
  runaway caller hits.

### Response when limited

```
HTTP/1.1 429 Too Many Requests
Retry-After: <seconds>
Content-Type: application/json

{
  "error": {
    "code": "rate_limited",
    "message": "Per-credential rate limit exceeded.",
    "request_id": "<id>",
    "details": { "retry_after_seconds": <seconds>, "limit_class": "recommend" }
  }
}
```

`Retry-After` is an integer number of seconds, computed by the Lua script
from the token deficit and the refill rate. The client can sleep exactly
that long and retry. The `details` object carries the class so a client
that talks to more than one endpoint can tell which bucket it filled.

`X-RateLimit-Limit` and `X-RateLimit-Remaining` are not returned. They
would require reading the bucket on every request (an extra round trip
for a header nobody has asked for). The `Retry-After` on the refusal is
the piece clients act on; the running count is not.

### Configuration

| Variable | Meaning | Default |
|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | Default limit for classes that do not override | 600 |
| `RATE_LIMIT_RECOMMEND_PER_MINUTE` | Limit for the `recommend` class | falls back to `RATE_LIMIT_PER_MINUTE` |
| `RATE_LIMIT_EVENTS_PER_MINUTE` | Limit for the `events` class | falls back to `RATE_LIMIT_PER_MINUTE` |
| `RATE_LIMIT_ADMIN_PER_MINUTE` | Limit for the `admin` class | falls back to `RATE_LIMIT_PER_MINUTE` |
| `RATE_LIMIT_BUCKET_MULTIPLIER` | Bucket size = limit_per_minute × this | 0.1 (a 600/min limit allows bursts of 60) |
| `INSTANCE_COUNT` | Number of running API instances, used to split the limit in the fallback | 1 |

The bucket multiplier expresses the burst/capacity trade-off. A limit of
600/min with a multiplier of 0.1 means the bucket holds 60 tokens: a
caller can burst 60 requests, then is smoothed to 10 per second. A
higher multiplier permits larger bursts; a lower one is stricter. It is
config because the right value depends on the caller mix, not on a
technical constraint.

### Observability

New metrics:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `recsys_rate_limit_hits_total` | counter | `class`, `result` | `result` ∈ `allowed` / `limited` |
| `recsys_rate_limit_remaining` | histogram | `class` | Tokens left in the bucket at decision time |
| `recsys_rate_limit_degraded` | gauge | — | 1 when the fallback limiter is active |
| `recsys_rate_limit_lua_errors_total` | counter | `type` | Lua script failures by category |

`recsys_rate_limit_remaining` is a histogram, not a gauge, for the same
reason the latency metric is a histogram: the per-request distribution is
what matters (is the bucket usually near empty?), not a single value.

Log lines (structlog, JSON):

- `ratelimit.allowed` at DEBUG (not INFO: one line per request is noise).
- `ratelimit.limited` at INFO, with `class`, `credential_hash_prefix`,
  `retry_after_seconds`.
- `ratelimit.degraded` at WARNING, once per process when the fallback
  activates, and once more when it deactivates.

## Alternatives considered

| Option | Why not |
|---|---|
| **Sliding window with a Redis sorted set** | O(N) memory per key and O(log N) work per request, plus a mandatory reap step. At the limits this project is sized for, the sorted sets are the most expensive state in Redis. The token bucket is O(1) with the same smoothing behavior for the caller-visible metric. |
| **Fixed window (one counter per minute)** | Simple, but the boundary is a cliff: a caller can send the full limit at 11:59:59 and again at 12:00:00. The token bucket smooths the same limit without a cliff. |
| **No rate limit** | A single runaway caller takes down the retrieval path for every other caller. Even a portfolio project should not ship that failure mode. |
| **Per-IP rate limit** | Wrong unit (see Context). A NAT hides many callers, a distributed caller hides behind many IPs, and an attacker with a fresh address pool is not limited at all. |
| **Fail-closed when Redis is down (reject every request)** | Turns a Redis outage into an API outage. The rate limiter is a safety mechanism, not an authorization decision (that is ADR-0013, which does fail-closed); a degraded limiter is worse than no limiter, but a refused request is worse than a slower one. |
| **Fallback keeps the full limit per instance** | Multiplies the effective limit by the instance count. The point of a shared limiter is the aggregate; a fallback that lets N× more traffic through fails the same caller whose behavior it exists to bound. |
| **Return `X-RateLimit-*` headers on every response** | Requires reading the bucket on the allowed path, which is one more Redis round trip per request for a header no client of this project has asked for. The refusal path (`Retry-After`) is what callers act on. |
| **Rate limit health checks** | A `429` on `/readyz` reads to an orchestrator as "not ready" and drains a healthy instance. The path is cheap and unauthenticated; the attack surface is not worth the metric. |
| **Shared limiter state in a database** | PostgreSQL is already on the hot path for retrieval; putting the limiter there as well means a slow query slows every request's admission. Redis is the right tier for the counter, and it is already a dependency (ADR-0015). |
| **Per-endpoint limits but no classes** | Every endpoint would have its own env var, and adding a route would require a config edit. Classes group routes by cost, which is the property the limit is actually about. |

## Consequences

**Positive**

- **Memory and CPU are bounded by the number of credentials, not the
  number of requests.** The bucket state per key is two numbers, no
  matter the traffic.
- **The refusal carries a real retry time.** `Retry-After` is computed
  from the deficit, so a client that obeys it lands inside the bucket on
  the first try. A fixed "retry in a minute" would either overshoot
  (wasting capacity) or undershoot (refusing again).
- **The failure mode is stated and observable.** Redis down does not
  stop traffic; it degrades to a per-instance limiter whose aggregate is
  close to the configured limit. The alert fires on the degradation, not
  on the outage, so the operator sees the actual change.
- **Health checks are exempt by name.** A future change that adds a rate
  limit class must exclude `/readyz` deliberately; it cannot happen by
  forgetting an entry.
- **The credential never appears in a key.** A Redis dump leaks hashes,
  not API keys or JWTs.

**Negative / accepted trade-offs**

- **The fallback is approximate.** `INSTANCE_COUNT / configured_limit`
  per instance is an average, not an exact partition. A caller whose
  requests happen to land evenly across instances is limited slightly
  more strictly than configured; one who lands on a single instance
  during a Redis outage is limited slightly less. The error is bounded by
  the instance count and is accepted for the availability it buys.
- **`INSTANCE_COUNT` is a config value, not a discovered one.** A
  misconfigured value is a silent change to the fallback limit. The
  correct fix is a service registry, which is more infrastructure than
  M3 needs; the value is logged at startup so a misconfiguration is
  visible in the startup line.
- **One Lua script is one more artifact to test.** It is exercised
  against a real Redis in `tests/integration/test_rate_limit.py` and is
  the same script in every environment; a version skew between
  environments is a deployment error, not a supported state.
- **The burst multiplier is a magic number.** 0.1 (a 600/min limit
  permits bursts of 60) is a starting point, not a derivation. It is
  config, and the README documents that it will be revised against
  observed behavior once the M4 load test measures real bursts.
- **Two limits (`recommend`, `events`) and one shared default is one
  more config surface.** The alternative — one limit for all classes —
  either throttles event ingestion to the recommendation cost or allows
  a recommendation flood at the event-ingestion rate. Neither is right.
- **No `X-RateLimit-Limit` / `-Remaining` headers.** Clients cannot
  self-pace before hitting the wall; they learn the limit from the
  refusal. This is acceptable for the project's callers (a small number
  of B2B integrations and a first-party client that reads the API docs).
  If a caller with a hard requirement for pre-emptive headers appears,
  a future ADR adds them, with the extra round trip priced in.

## References

- `docs/adr/0013-authentication-strategy.md` — the credential this
  limiter keys on
- `docs/adr/0015-cache-and-circuit-breaker.md` — the Redis dependency
  this limiter shares
- `docs/contracts.md` § 2.5 — the `429` response shape
- `docs/contracts.md` § 3 — the rate limit config variables
- `src/recsys/api/middleware/rate_limit.py` — the middleware that
  executes the Lua script
- `tests/integration/test_rate_limit.py` — the tests against a real
  Redis, including the fallback path
