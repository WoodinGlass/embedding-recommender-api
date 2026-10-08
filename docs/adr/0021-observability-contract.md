# ADR-0021: Observability contract (metrics, logs, traces)

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

M0 established three observability primitives:

- **structlog JSON logs** (`src/recsys/monitoring/logging.py`), with
  `request_id` bound per request.
- **A Prometheus registry** (`src/recsys/monitoring/metrics.py`) with a
  fixed set of metric names and labels (`docs/contracts.md` § 4.1).
- **An empty tracing module** (`src/recsys/monitoring/tracing.py`) that
  is a no-op until "M4".

`docs/contracts.md` § 4 reserves field and metric names but does not say
how the three primitives agree with each other, what the cardinality
budget is, or how a reviewer moves from a spike in a metric to the log
lines that explain it. Without those rules, three common failures appear
as the code grows:

1. **A metric label explodes the time series count.** Someone adds
   `user_id` or `request_id` to a Prometheus label, and the metric's
   cardinality grows with traffic. Prometheus's scrape, its storage, and
   every query against the metric slow down together. The problem is
   invisible in development (few users) and catastrophic in production.
2. **A log line and a metric cannot be joined.** A metric alert fires
   for a route and a status; the log lines for those requests carry a
   `request_id` the alert does not know. A reviewer greps by time window
   and hopes. The join key was available when the metric was recorded
   and was not written.
3. **The logs and the traces tell different stories.** A log line says
   `source=ann`; a trace says the request spent 80 ms in the cache and
   4 ms in the ANN. Both are true at different moments; without a shared
   identifier and a shared schema, a reviewer cannot tell which moment
   each describes.

M3 makes the API serve real requests, so the observability contract has
to be fixed now, before the code multiplies the surface.

## Decision

### One correlation id, propagated in three places

Every request has exactly one id, generated or accepted by
`RequestContextMiddleware` (M0). It is:

- **Returned** in the `X-Request-ID` response header.
- **Bound** to structlog's contextvars so every log line written during
  the request carries it as `request_id`.
- **Attached** to every span in the request as `request.id`.

The value is either the client's `X-Request-ID` (truncated to 64
characters, bounded) or a server-generated 16-hex-character string. A
client that supplies its own id can correlate its logs with the
server's without a separate lookup.

**The request id is not the trace id.** A trace id (from OpenTelemetry)
identifies a distributed trace; a request id identifies one HTTP
request. In M3 they are distinct: the trace id comes from the tracing
SDK's context, the request id from the middleware. Both appear in the
logs (`trace_id`, `request_id`) so a reviewer can pivot on either.
Collapsing them into one value would be a mistake the day a request
makes an outbound call that belongs to the same trace but is a different
request.

### Metrics: names, labels, and cardinality budget

`docs/contracts.md` § 4.1 is the source of truth for metric names. This
ADR adds the rules for what may appear in a label.

**A label value must be drawn from a bounded set known at build time.**
Concretely:

| Allowed | Forbidden |
|---|---|
| Route template (`/v1/items/{item_id}/similar`) | Resolved path (`/v1/items/i_789/similar`) |
| Enum from a closed set (`cache` / `ann` / `fallback_ann` / `fallback_cached`) | Free-form string (`source: "the-cache"`) |
| HTTP method (`GET`, `POST`) | Request body fragment |
| HTTP status class (`2xx`, `4xx`, `5xx`) or exact code from a fixed set | — |
| Error category from `ErrorCode` (`auth`, `validation`, `retrieval`, `cache`, `upstream`, `internal`) | Exception class name (`psycopg.errors.UniqueViolation`) |
| Experiment name and variant, both declared in `experiments.yaml` | `user_id`, `item_id`, `request_id`, `trace_id`, `event_id` |
| Index version (`idx-a3f9e021`); bounded by the registry | Any value the caller can set |

**Why the boundary.** Prometheus stores one time series per unique label
combination. A label whose values grow with traffic makes the metric's
cardinality grow with traffic, and the cost is paid on every scrape,
every query, and every alert evaluation. A metric with `user_id` on a
service with a million users has a million time series for that metric;
the storage is the user count times the samples per minute times the
retention window, and it is the single most common way a Prometheus
deployment falls over.

**The forbidden list is mechanical.** A value that comes from a request
body, a path parameter, a query parameter, or a user identity is
forbidden as a label. A value that comes from a config enum, a
registered index version, or a route template is allowed. The rule is
"would this value be bounded if traffic doubled?"; if the answer is no,
it is a log field, not a label.

**The error category mapping is closed.** The `recsys_errors_total{type}`
metric's `type` label is one of a fixed set defined in the code
(`ErrorCode` plus a few internal categories). A `try/except` that
records `type=type(exc).__name__` would create a new time series per
exception class, including any third-party exception that happens to be
caught. The mapping is a function `exception -> category` in one module;
adding a category is a code change with a test that the mapping's output
set is closed.

### New metrics for M3

`docs/contracts.md` § 4.1 is updated in the same PR as the code. The
additions, each with its bounded label set:

| Metric | Type | Labels | Emitted where |
|---|---|---|---|
| `recsys_cache_requests_total` | counter | `result` ∈ `hit`/`miss`/`bypass`/`error` | Cache middleware |
| `recsys_cache_negative_hits_total` | counter | — | Cache middleware |
| `recsys_cache_write_errors_total` | counter | `type` (closed set) | Cache middleware |
| `recsys_fallback_total` | counter | `tier` ∈ `fallback_ann`/`fallback_cached` | Recommend handler |
| `recsys_rerank_duration_seconds` | histogram | `arm` (from `experiments.yaml`) | Re-ranker |
| `recsys_rerank_failures_total` | counter | `step` ∈ `normalize`/`blend`/`mmr`, `type` (closed set) | Re-ranker |
| `recsys_rerank_skipped_total` | counter | `reason` ∈ `no_vectors`/`k_below_threshold` | Re-ranker |
| `recsys_rate_limit_hits_total` | counter | `class`, `result` ∈ `allowed`/`limited` | Rate limit middleware |
| `recsys_rate_limit_remaining` | histogram | `class` | Rate limit middleware |
| `recsys_rate_limit_degraded` | gauge | — | Rate limit middleware |
| `recsys_rate_limit_lua_errors_total` | counter | `type` (closed set) | Rate limit middleware |
| `recsys_circuit_breaker_state` | gauge | `name` ∈ `redis`/`postgres` | Breaker |
| `recsys_circuit_breaker_trips_total` | counter | `name`, `reason` ∈ `failures`/`timeouts` | Breaker |
| `recsys_experiment_exposures_total` | counter | `experiment`, `variant` (both from `experiments.yaml`) | Exposure writer |
| `recsys_experiment_exposure_errors_total` | counter | `type` (closed set) | Exposure writer |
| `recsys_readyz_status` | gauge | `status` ∈ `ready`/`degraded`/`not_ready` | Readiness |
| `recsys_readyz_check_duration_seconds` | histogram | `check` ∈ `db`/`index`/`redis` | Readiness |

The two `experiments.yaml` labels (`experiment`, `variant`) are bounded
by the file's contents, which is committed. Adding an experiment adds
two time series per metric; the file's growth is bounded by human
review.

### Logs: structured, versioned, and reserved

`docs/contracts.md` § 4.2 is the log field contract. This ADR adds:

**A `schema_version` field to every log line.** Value `1` in M3. The
field is what a log consumer keys on to know which set of reserved
fields to expect. A future change to the log schema increments the
version; a consumer that reads `schema_version=1` lines can ignore
`schema_version=2` lines or route them elsewhere. Without the field, a
schema change is a silent break for every consumer.

**A fixed `event` naming convention.** `<subsystem>.<action>`, with
lowercase and dots. Examples: `http.request`, `cache.hit`, `cache.miss`,
`retrieval.fallback.ann`, `rerank.mmr.skipped`, `index.build.done`. A
reviewer can grep by prefix (`cache.`) to find every cache-related line.
The convention is enforced by review, not by a linter, but the set of
prefixes is small (`http`, `cache`, `retrieval`, `rerank`, `index`,
`rate_limit`, `experiment`, `readyz`, `pipeline`, `bench`).

**Reserved fields** (from `docs/contracts.md` § 4.2, restated here with
the M3 additions):

- `schema_version`, `ts`, `level`, `event` — always.
- `request_id`, `trace_id`, `span_id` — when in a request context.
- `route`, `method`, `status`, `duration_ms`, `source` — on request
  lines.
- `user_id_hash` — when the log line is user-scoped, and **never** the
  raw `user_id`.
- `experiment`, `variant`, `assignment_bucket` — on experiment-related
  lines.
- `error.type`, `error.message` — on errors; `message` is sanitized
  (no raw exception repr if it contains a credential; the exception
  category is in `error.type`).

**Forbidden fields:**

- `user_id`, `item_id`, `event_id`, `request_body`, `api_key`,
  `jwt`, `authorization`, `password`, `token`, `secret`,
  `database_url`, `redis_url`, `salt`, `onnx_vector`.

The list is a rule, not a suggestion. A test in
`tests/unit/test_log_redaction.py` runs the recommend path with sentinel
values in each of these fields' natural locations (a `user_id` header,
an `X-API-Key`, a request body) and asserts that none of the sentinels
appear in captured stdout or stderr.

### Trace: one span per stage, no external collector required

The tracing module is no longer a no-op. It emits spans, and it has two
modes:

- **In-process only** (default in `dev` and in CI): spans are recorded
  in a ring buffer the process holds, and are exposed at `/debug/traces`
  when `APP_ENV=dev`. The ring buffer is bounded (last 1 000 requests)
  and the endpoint is disabled outside `dev`. This is enough to
  correlate a slow request's stages during development without running
  a collector.
- **OTLP export** (opt-in): when `OTEL_EXPORTER_OTLP_ENDPOINT` is set,
  spans are exported to an OpenTelemetry collector. In `prod` this is
  the expected mode; in dev it is optional.

**Span names** (from `docs/contracts.md` § 4.3, restated):

| Span | Attributes |
|---|---|
| `http.server.request` | auto-instrumented; carries `request_id` |
| `recsys.auth` | `principal.kind` (`api_key`/`jwt`), `scopes.count` |
| `recsys.rate_limit` | `class`, `result`, `remaining` |
| `recsys.cache.get` | `cache.key.kind`, `cache.result` |
| `recsys.retrieval.search` | `index.version`, `filters.count`, `k` |
| `recsys.rerank` | `rerank.arm`, `rerank.mmr.applied` |
| `recsys.fallback` | `fallback.tier`, `fallback.reason` |
| `recsys.exposure.write` | `experiment`, `variant` |

Attributes follow the same bounded-set rule as metric labels. A span
attribute that would be a forbidden metric label is also a forbidden
span attribute (a collector can turn a span attribute into a metric
label; the rule is the same).

**Sampling:** 100% of error traces (any span with `status=error`), 10%
of success traces in `prod`, 100% in `dev` and `staging`. The ratio is
config (`OTEL_TRACES_SAMPLER_ARG`). The reason for two ratios: errors
are the traces a reviewer needs and there are few of them; successes
are the traces that produce volume, and a sample is enough to see the
latency distribution.

### The three primitives agree on the request id

A reviewer's path from a metric alert to the log lines and the trace:

1. The alert names a metric (`recsys_request_duration_seconds_p95`) and
   a label set (`route=/v1/recommend`, `source=ann`).
2. The reviewer queries the metric's histogram buckets in the alert's
   time window and finds the individual request ids by querying the
   logs: `event=http.request AND route=/v1/recommend AND source=ann AND
   duration_ms>200` over the same window.
3. The log lines carry `request_id` and `trace_id`. Either pivots into
   the trace (via the OTLP backend or the dev ring buffer) or into the
   rest of the logs for that request.

The metric carries only the bounded labels; the request-level detail
lives in the logs and the trace. This is the design: metrics for the
aggregate, logs for the individual request, traces for the stages of
one request.

**Do not put `request_id` on a metric.** A metric label with one value
per request is the cardinality explosion the forbidden list exists to
prevent. The join key is carried on the log line (and the trace), not on
the metric.

### The `/metrics` endpoint is not authenticated but is not public

`docs/contracts.md` § 2 marks `/metrics` as "Internal". In M3 it is
reachable without authentication (Prometheus's scraper does not carry
the API's credentials), but it should not be exposed to the public
internet. The deployment documentation (M6) puts it on a network
reachable only by the scraper; the code does not enforce a network
boundary it cannot see.

The rationale for the mark: the metric labels are bounded and do not
carry user data or credentials, so an unauthenticated read of `/metrics`
is not a PII leak. It is, however, an internal surface that a public
deployment should not expose. The decision is deployment-side and is
recorded here so the intent is not lost.

## Alternatives considered

| Option | Why not |
|---|---|
| **One id for both request and trace** | A request id identifies one HTTP request; a trace id identifies a distributed trace that may span multiple requests. Collapsing them is a mistake the day an outbound call belongs to the same trace but is a different request. Both fields appear in the logs; a reviewer pivots on either. |
| **Add `request_id` to a metric label** | One time series per request. The cardinality explosion is the exact failure the forbidden list exists to prevent. The join key goes on the log line. |
| **Error label = `type(exc).__name__`** | A new time series per exception class, including third-party exceptions. The `ErrorCode` mapping is a closed set; a new category is a code change with a test that the set is closed. |
| **No `schema_version` on log lines** | A schema change silently breaks every log consumer. The version is one small field and one class of future incident avoided. |
| **Traces require a collector in every environment** | Makes the developer's local run depend on a collector they do not have. The in-process ring buffer covers development; the OTLP exporter covers production. Two modes, one span API. |
| **Sampling every trace** | Volume grows with traffic for no new signal. The two-ratio policy (100% errors, 10% successes) keeps the signal without the volume. |
| **Sampling by request id (a stable hash)** | A trace that is sampled would be sampled across all its spans; that is what the SDK's sampler does. A hand-rolled per-request sampler would need to propagate the decision, which is what the SDK already does. Not reinvented here. |
| **A separate log schema for request-level and pipeline-level lines** | Two schemas, two consumer implementations, one shared reserved-field list. The `schema_version` field covers both without a fork. |
| **Auth on `/metrics`** | Prometheus's scraper does not carry the API's credentials. Adding auth means adding a scraper configuration (a bearer token or mTLS). The labels are bounded and carry no secrets; the correct boundary is the network, which is a deployment concern. |
| **A `debug.trace` field on every response** | Exposes internal stage timings to every client. The dev-only `/debug/traces` endpoint covers the debugging use case; a production response carries `request_id` and the client (or the operator) pivots through the log. |
| **Log every request at INFO with full detail** | The access log line (`http.request`) is one line per request; adding per-stage INFO lines multiplies the volume by the number of stages. Per-stage detail goes on spans (sampled) and on DEBUG log lines (not collected in production by default). |
| **Prometheus remote-write to a hosted service in M3** | Adds a paid dependency and a network hop for a single-node project whose metrics are read by a local Grafana (M4). Remote-write is a future ADR if the dashboards move off the local Prometheus. |

## Consequences

**Positive**

- **Cardinality is bounded by design, not by discipline.** The label
  rules make a `user_id` label a review-time error rather than a
  production incident.
- **A metric alert is one query away from the log lines that explain
  it.** The request id is carried on the log line, and the metric's
  bounded labels are what the query filters on.
- **Traces work in development without a collector.** The dev-mode ring
  buffer and `/debug/traces` endpoint cover the local case; the OTLP
  exporter is opt-in.
- **The log schema is versioned.** A future change to the reserved
  fields is a `schema_version` bump, and a consumer keyed on the version
  can ignore or route lines it does not understand.
- **The forbidden-field list has a test.** `test_log_redaction.py` runs
  the recommend path with sentinel values and asserts they do not appear
  in captured output. The rule is enforced by CI, not by memory.

**Negative / accepted trade-offs**

- **More metrics than M0.** The additions above roughly double the
  registry. Each is bounded and has a reason; the alternative is a
  single generic metric with a `stage` label, which would lose the
  per-stage type (a histogram's buckets differ from a counter's
  increment).
- **Two tracing modes.** More code than a single mode would be. The
  in-process mode is what makes development workable; the OTLP mode is
  what production needs. Both are small.
- **`/debug/traces` is a new endpoint to secure.** It is disabled
  outside `dev` by a check on `APP_ENV`; a deployment that sets
  `APP_ENV=dev` in production has a problem the endpoint will not be
  the only symptom of. The check is a guard, not a security boundary.
- **Sampling successes at 10% means some slow requests are not traced.**
  The metric's histogram and the access log's `duration_ms` cover every
  request; the trace covers a sample. A reviewer who needs the trace of
  a specific slow request has a 10% chance the trace exists and a 100%
  chance the log lines do. The trade-off is volume.
- **Log lines at DEBUG are not collected in production by default.**
  `LOG_LEVEL` is INFO in production; DEBUG lines (`cache.hit`,
  `cache.miss`) are not emitted. The aggregate is in the metrics; the
  per-request detail at INFO is the access log. A reviewer who needs
  DEBUG detail raises the level for a window, which is a deliberate
  operational action, not a default.
- **The forbidden-field list is long.** Twelve fields to remember. The
  test enforces the list; a field added to the code but not the list is
  a review miss the test would not catch. The list is the contract; the
  test is a check.

## References

- `docs/contracts.md` § 4 — the metric, log, and span contracts this
  ADR extends
- `docs/adr/0009-golden-set-and-metrics.md` — the offline metrics whose
  names this ADR does not change
- `docs/adr/0015-cache-and-circuit-breaker.md` — the cache and breaker
  metrics this ADR lists
- `docs/adr/0017-experiment-assignment.md` — the bounded experiment and
  variant labels
- `docs/adr/0019-readiness-contract.md` — the readiness metrics
- `docs/adr/0020-fallback-chain.md` — the fallback tier metric
- `src/recsys/monitoring/logging.py` — the log schema and
  `schema_version`
- `src/recsys/monitoring/metrics.py` — the registry
- `src/recsys/monitoring/tracing.py` — the two-mode tracer
- `tests/unit/test_log_redaction.py` — the forbidden-field test
- `tests/unit/test_metric_contract.py` — the guard that metric names
  match `docs/contracts.md`
