# Runbook

> **Status:** skeleton. Alerts and dashboards are wired in M4; the procedures
> below are written now so that the on-call response is designed alongside the
> system, not after the first incident.

## Service overview

`embedding-recommender-api` serves embedding-based item recommendations over
an ANN index, with a popularity fallback when retrieval is unavailable. It
depends on PostgreSQL 16 (pgvector) for catalog and embeddings, and Redis for
response caching. Both are optional at the request level: the service degrades
rather than failing (see `docs/decisions.md` § 6).

**SLOs (targets, pending measurement in M4):**

| Metric | Target |
|---|---|
| Availability of `POST /v1/recommend` | 99.5% (30-day) |
| Latency p95 at target RPS | < 200 ms |
| Fallback rate | < 1% of requests |
| Error rate (5xx) | < 0.1% of requests |

## First response

1. **Check `/readyz`** on the affected instance. A non-200 means the instance
   should be drained by the load balancer; the incident may already be
   self-healing on the remaining instances.
2. **Check `/metrics`** for `recsys_active_index_info` and
   `recsys_fallback_total{reason=...}`. The `reason` label names the failure.
3. **Read the last 100 structured log lines** filtered by `level=error` or
   `event=http.request` with `status>=500`. Every line carries `request_id`
   and `trace_id` for correlation.

## Runbook 1 — `/readyz` fails (DB unreachable or index not loaded)

**Symptom:** readiness probe fails, instance removed from rotation, 503s to
clients. `recsys_fallback_total{reason="db_down"}` or
`{reason="index_missing"}` increases.

**Diagnose**

```bash
curl -fsS http://<instance>:8000/readyz
# Expect: {"status": "ready", "checks": {...}} — inspect `checks` for the
# specific component that reports false.

psql "$DATABASE_URL" -c "SELECT 1"           # DB reachable?
psql "$DATABASE_URL" -c "SELECT extname FROM pg_extension WHERE extname='vector'"
curl -fsS http://<instance>:8000/metrics | grep recsys_active_index_info
```

**Mitigate**

- **DB unreachable:** restart the database or promote a standby. The service
  serves the popularity fallback until the DB is back; no action is required
  on the API instances.
- **Index not loaded but DB is up:** re-run `make index-promote
  VERSION=<last-known-good>`. If the active index is corrupt, roll back with
  `make index-rollback`, which flips the pointer to the previous version.

**TODO (M3):** `/readyz` currently reports ready unconditionally; the DB and
index checks are added when retrieval lands.

## Runbook 2 — p95 latency above 200 ms for 10 minutes

**Symptom:** alert `HighP95Latency`. `histogram_quantile(0.95, ...)` over
`recsys_request_duration_seconds{route="/v1/recommend"}` exceeds 0.2.

**Diagnose** (in order — most likely cause first)

1. **Cache hit rate drop.** `rate(recsys_cache_requests_total{result="hit"})`
   vs. `{result="miss"}`. A sudden drop points to Redis; a gradual drop points
   to a TTL or key-composition change.
2. **ANN query latency.** Check the `ann` span duration. If ANN queries are
   the bottleneck, look at `HNSW_EF_SEARCH` — a recent increase raises both
   recall and latency.
3. **DB contention.** Check PostgreSQL slow-query log and `pg_stat_activity`
   for long-running queries.
4. **Traffic spike without proportional cache warmup.** Cold cache after a
   deploy produces a temporary p95 spike; check whether the deploy timestamp
   matches the alert window.

**Mitigate**

- **Lower `HNSW_EF_SEARCH`** (e.g. 100 → 60) as a temporary measure. This
  reduces recall; record the change and revert once the incident is over.
- **Increase `CACHE_TTL_SECONDS`** to raise hit rate. Bounds staleness;
  acceptable during an incident.
- **Scale horizontally** if the instance is CPU-saturated and the DB is not.

**TODO (M4):** the metrics and dashboards referenced above are populated when
Prometheus and Grafana are wired.

## Runbook 3 — Fallback rate spike

**Symptom:** alert on `rate(recsys_fallback_total[5m])`. Requests are served
by the popularity fallback, which is a correctness-preserving but
relevance-degrading path.

**Diagnose**

Break down by `reason`:

| `reason` | Likely cause |
|---|---|
| `db_down` | PostgreSQL unreachable; check `psql "$DATABASE_URL" -c "SELECT 1"` |
| `ann_timeout` | ANN query exceeds `ANN_TIMEOUT_MS`; check `HNSW_EF_SEARCH` and DB load |
| `cold_start` | Expected for new users without seed items; verify rate is within baseline |
| `index_missing` | Active index pointer not loaded on some instances; check `/readyz` on each |

**Mitigate**

- **`db_down`:** see Runbook 1. The service remains available; the incident is
  a degradation, not an outage.
- **`ann_timeout`:** temporarily raise `ANN_TIMEOUT_MS` (accepting the latency
  cost) or lower `HNSW_EF_SEARCH` (accepting the recall cost). Record which
  was chosen and why.
- **`index_missing`:** re-promote or roll back the index (Runbook 1).
- **`cold_start` above baseline:** this is a data problem, not an
  infrastructure problem. Check whether seed-item ingestion has stopped.

## Runbook 4 — SRM detected in a running experiment

**Symptom:** the SRM check in `make ab-analyze` reports
`p_value < 0.001` on the observed variant assignment counts.

**This is a validity failure, not a performance failure.** Any result from the
experiment is uninterpretable until the SRM is resolved.

**Diagnose**

1. Re-run the assignment function on a sample of `user_id`s and compare to
   the logged `variant`. A mismatch means the salt changed mid-experiment.
2. Check for users assigned but never logged (logging gap) and users logged
   but never assigned (double logging).
3. Check for a variant-specific error that prevents exposure logging in one
   arm.

**Mitigate**

- If the salt changed: **stop the experiment**. Results are not salvageable.
- If the logging gap is one-sided: fix the logging, then restart the
  experiment with a fresh salt. Do not "correct" by re-weighting.
- Record the incident in the experiment's metadata so the decision log
  reflects that the run was invalidated.

**TODO (M5):** assignment and logging land with the A/B feature. This runbook
is a design artifact until then.

## Rollback

| What | How | Time |
|---|---|---|
| Code | Redeploy the previous image tag (CD does this automatically on failed readiness). | ~1–2 min |
| Index | `make index-rollback` — flips the active-version pointer. | < 5 s |
| Migration | `alembic downgrade -1` — only after confirming the new schema is not required by the running code. | seconds |

Code and index rollbacks are **independent**. Rolling back code does not
require rolling back the index, and vice versa. This is by design (see
`docs/decisions.md` § 3, blue/green index swap).

## Escalation

Solo project: the maintainer is the escalation path. The template below is
kept so that the escalation model is explicit for future collaborators.

```
Severity 1 (service down):        page maintainer immediately
Severity 2 (degraded, no outage): notify maintainer, respond within 1h
Severity 3 (metric anomaly):      open an issue, respond next business day
```

## TODO markers

| Section | Landed in |
|---|---|
| `/readyz` real checks | M3 |
| Prometheus alerts and Grafana dashboards | M4 |
| A/B assignment, logging, and SRM analysis | M5 |
| CD, staging→prod, automatic rollback | M6 |
| Churn endpoint monitoring | M7 |
