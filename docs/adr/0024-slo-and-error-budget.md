# ADR-0024: SLOs, error budget, and what this project does not promise

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

Three milestones depend on a definition of "the system is working":

- **M4's load test** needs a pass/fail criterion for the served path. The
  README names "p95 < 200 ms at the target RPS" as a target, but a target
  without a window and a definition of the population is not a number a
  test can check.
- **M5's A/B guardrails** compare a variant's reliability metrics against
  the system's; the comparison needs a baseline with a stated meaning.
- **M6's CD auto-rollback** fires when the deployed image breaches
  "readiness or SLO checks". Readiness is defined (ADR-0019). The SLO
  side is not.

The M0 scaffold promises nothing formal. `docs/runbook.md` lists "SLOs
(targets, pending measurement in M4)" in a table with four rows and no
definitions. That table is a placeholder; this ADR is the contract
behind it.

Three questions the project has to answer before writing code that
consumes these numbers:

1. **Is there an SLA?** An SLA is a promise to a customer with a
   remedy (a refund, a credit). This project has no customers and no
   contract; an SLA would be a fictional promise.
2. **What are the SLOs?** Internal targets, in the sense that they are
   the thresholds the system's behavior is measured against.
3. **What is the error budget, and what happens when it is spent?**
   An error budget without a policy is a number nobody acts on.

## Decision

### No SLA

This project has no external SLA. There is no customer, no contract,
no financial remedy, and no uptime commitment. The SLOs below are
internal targets. If a future milestone deploys the service for real
users, an ADR adds the SLA and the numbers here inform its design;
until then, writing an SLA would be a promise the project cannot keep.

The distinction matters for a reviewer: the SLOs are enforced (by the
load test in M4, by the alerts in M4, by the CD checks in M6), and the
absence of an SLA is honest about the project's scope.

### Three SLOs

Each SLO is a target for a metric, over a window, with a definition of
"good" and "bad" events. The three are:

| SLO | Target | Window | Good event | Bad event |
|---|---|---|---|---|
| **Availability** | ≥ 99.5% | rolling 30 days | A request whose response status is not 5xx | A request whose response status is 5xx |
| **Latency** | ≥ 95% under 200 ms | rolling 30 days | A request whose `duration_ms` ≤ `LATENCY_SLO_MS` | A request whose `duration_ms` > `LATENCY_SLO_MS` |
| **Quality** | ≤ 1% fallback | rolling 30 days | A request whose `meta.source` is `cache` or `ann` | A request whose `meta.source` is `fallback_ann` or `fallback_cached` |

`LATENCY_SLO_MS` is config (`LATENCY_SLO_MS`, default 200), the same
number the README names. The rollout target RPS (`LATENCY_SLO_RPS`) is
set in M4 against the measured capacity; the latency SLO applies at
that RPS, not at arbitrary load.

### The population

"Request" means a request to an API route, excluding the infrastructure
endpoints:

- **Excluded:** `/livez`, `/healthz`, `/readyz`, `/metrics`. These are
  probe and scrape paths; including them would measure the monitoring
  system, not the service.
- **Included:** every `/v1/*` route.

A request that arrives while the service is under a planned maintenance
window is excluded from the population (the window is declared in the
deployment and recorded in the incident log). Planned maintenance is
rare for this project; the exclusion exists so the window is a declared
event, not a gap in the data.

### Availability: 4xx is not a failure

The availability SLO counts 5xx as failures and everything else
(2xx, 3xx, 4xx) as successes. The reasons:

- **4xx is the system working.** A `422` for a malformed request is the
  API answering correctly. Counting it as an availability failure would
  make the SLO a function of client behavior, not of the service.
- **The project needs a separate 4xx signal.** A spike in 401s may mean
  a broken client, an expired key, or an attack. That signal is a
  metric (`recsys_requests_total{status="401"}`), not an SLO. Adding a
  fourth SLO for "rejects requests correctly" would measure a quantity
  that is not the system's fault.
- **429 is a special case.** A rate-limited request is the rate limiter
  working (ADR-0014). It counts as a success for availability. A
  sustained 429 rate is a capacity or abuse signal, tracked separately;
  it does not consume the availability budget.

### Latency: 5xx requests do not count against the latency SLO

A request that failed with a 5xx was not "slow" in the sense the SLO
measures; it failed. Counting it against latency (as if it had taken a
long time) would conflate two different failure modes and would make
the latency SLO move when availability moves, for no reason the operator
can act on.

Similarly, a rate-limited request (`429`) has a fast response by
construction (the limiter refused before doing work) and does not
represent the retrieval path's latency. It is excluded from the
population for the latency SLO.

### Quality: the fallback rate is its own SLO

A response served from `fallback_ann` or `fallback_cached` (ADR-0020)
has a 200 status and is a valid response. It is not an availability
failure. It is a quality degradation: the user got a popular list
instead of a personalized one.

The quality SLO captures that: 1% of requests may be served from the
fallback over a 30-day window. Above that, the fallback is a systemic
condition worth an incident.

**Why 1%.** It is the same threshold the fallback-rate alert in
`docs/runbook.md` uses. A fallback rate below 1% is a database hiccup
that lasted a few seconds; above 1% is a condition that persists long
enough to matter. The number is a judgment, recorded here so it is a
stated trade-off rather than a constant in an alert rule.

### Error budget

For each SLO, the error budget is the complement of the target over the
window:

| SLO | Budget | In minutes (30 days) |
|---|---|---|
| Availability (99.5%) | 0.5% | 216 minutes (~3.6 hours) |
| Latency (95%) | 5% | 2160 minutes (~36 hours) |
| Quality (99%) | 1% | 432 minutes (~7.2 hours) |

The budgets are independent. Consuming the availability budget does not
consume the latency budget; a fast 5xx consumes one, a slow 2xx consumes
the other.

### Error budget policy

The policy is stated for two conditions. It is deliberately short
because a long policy is one nobody reads under pressure.

**At 50% of the availability budget consumed in a 7-day window**, or
**100% consumed in a 30-day window**:

- Non-critical changes (features, refactors, optimizations) are frozen.
  Bug fixes and reliability work continue.
- A postmortem entry is required in `docs/postmortems/` describing what
  consumed the budget. The entry is a text file with a date, the
  incident, and what changed as a result.
- The alert that fires at these thresholds is configured in M4; the
  policy is what the alert's name means.

**The freeze lifts** when the 30-day rolling budget is below 50% again,
which for a fast-burning incident is roughly 15 days after it stopped.

For the latency and quality SLOs, the same thresholds apply. The
policy is the same shape for all three because the shape is the point:
a metric that consumes the budget triggers a review, and the review is
where the decision lives.

**A solo project's enforcement is honor-system.** There is no process
that blocks a commit when the budget is exhausted. What exists is the
metric, the alert, and this document. The discipline is the maintainer's;
the mechanism is what makes the discipline visible to a future
collaborator.

### Fast and slow burn alerts

The M4 dashboards implement the Google SRE two-window pattern:

| Alert | Condition | Severity |
|---|---|---|
| Fast burn | Budget burn rate > 14.4× over 1 hour | page |
| Slow burn | Budget burn rate > 6× over 6 hours | ticket |
| Very slow burn | Budget burn rate > 3× over 3 days | dashboard note |

The burn rate is the fraction of the budget consumed per unit time,
normalized to the window. A rate of 1× consumes the budget exactly at
the window's end; 14.4× consumes it in about 2 hours (1/14.4 of 30
days).

The specific constants are the SRE workbook's defaults, not derived for
this project. They are config (`SLO_FAST_BURN_THRESHOLD`,
`SLO_SLOW_BURN_THRESHOLD`); the defaults are recorded here so a
reviewer can see the source.

### The SLO is not a gate for the offline evaluation

The offline evaluation (ADR-0010) measures retrieval quality against a
golden set. That is a different question from "is the served path
healthy". A retrieval metric below its threshold fails the eval gate; a
latency metric above its SLO fires the latency alert. The two are
independent, and this ADR does not touch the eval gate's thresholds.

The A/B guardrails (M5) use the SLOs as one input. A variant whose
p95 is above the SLO for the experiment's duration is flagged
regardless of its primary-metric lift. The mechanism is M5's; the SLO
is the number it compares against.

### Where the SLOs are defined

- **Availability and latency** are computed from
  `recsys_requests_total` and `recsys_request_duration_seconds`
  (ADR-0021). No new metric is added.
- **Quality** is computed from `recsys_fallback_total{tier}` (ADR-0021).
- The SLO definitions (target, window) live in
  `dashboards/slo.yaml` (M4), which the Grafana dashboards and the
  recording rules read. This ADR is the reasoning; the file is the
  configuration.

### What is not in M3

- **An SLA.** See above.
- **Per-route SLOs.** `/v1/recommend` and `/v1/events` have different
  latency profiles. M3 uses a single SLO per metric class; a future
  ADR splits them if M4's measurement shows the difference is material.
- **An SLO for the embedding pipeline.** The pipeline runs off the
  request path; a failed run is a data problem, not an availability
  problem. Its quality is measured by the offline evaluation gate
  (ADR-0010), not by an SLO.
- **An SLO for the churn endpoint.** The extension (M7) can add one;
  M3's three SLOs cover the retrieval service.

## Alternatives considered

| Option | Why not |
|---|---|
| **No SLOs; rely on the load test's numbers** | A load test produces a number; an SLO gives the number a meaning ("this is what we promise, this is how we measure it, this is what happens when we miss"). The load test's pass/fail criterion is the SLO. |
| **An SLA with a financial remedy** | There is no customer and no contract. Writing an SLA would be a fictional promise; the honest statement is the absence. |
| **99.9% availability** | Three nines on a single VPS with no redundancy is a number the deployment cannot deliver (a 43-minute monthly budget is less than a single database restart plus a redeploy). A target the system cannot meet is worse than a lower target it can. |
| **99% availability** | Two nines is a number a single-node deployment can meet but is low for a portfolio piece. 99.5% is the middle: honest about the deployment, meaningful as a target. |
| **Calendar month window** | A rolling 30-day window is smoother and does not have a "first of the month resets everything" cliff. The cliff is a real behavior (a bad end-of-month day consumes the whole month's budget); the rolling window avoids it. |
| **Count 4xx as availability failures** | Makes the SLO a function of client behavior. A broken client reduces the SLO the service is measured against, which is not a signal the operator can act on. |
| **Count 429 as availability failures** | A rate-limited request is the limiter working; the service is doing what it was configured to do. A sustained 429 rate is a capacity signal, tracked separately. |
| **Count 5xx against the latency SLO** | Conflates two failure modes. A fast 5xx is not "slow"; counting it against latency makes the two metrics move together for reasons the operator cannot separate. |
| **No latency SLO (availability only)** | A service that is up but slow is failing its users. The README already names a latency target; without an SLO it is a README sentence, not a threshold the CD pipeline can check. |
| **No quality SLO** | A fallback rate that grows from 0.1% to 5% is a degradation the availability and latency SLOs do not see (both metrics are fine during a database outage the fallback absorbs). The quality SLO is the one that catches it. |
| **One combined SLO** | The three signals have different bad-event definitions (a 5xx, a slow 2xx, a fallback 2xx). A combined metric would have to pick one definition and lose the others. |
| **Error budget policy with more thresholds** | The two thresholds (50% in 7 days, 100% in 30 days) are the SRE workbook's defaults; adding more makes the policy harder to remember without a clear benefit. |
| **Automatic freeze (a CI job that blocks merges)** | The mechanism would be a bot that reads Prometheus and fails a check; that is a component the project does not have, and for a solo project the discipline is the same either way. The alert and the document are the mechanism. |
| **Per-route SLOs in M3** | The difference between `/recommend` and `/events` is real but the project has one route class's worth of data. Splitting now would produce two under-measured SLOs; M4's load test informs the split if it is needed. |
| **SLO for the pipeline** | A pipeline failure is a data problem, not a service availability problem. The eval gate (ADR-0010) is the right mechanism for pipeline quality. |

## Consequences

**Positive**

- **M4's load test has a pass/fail criterion.** The latency SLO at the
  target RPS is the criterion; the load test's output is compared to
  the same number the alerts use.
- **The CD pipeline (M6) has an SLO check.** "Readiness or SLO checks"
  in the pipeline's description is defined: the availability SLO's
  burn rate over the rollout window.
- **The error budget policy is one paragraph.** A reviewer can read it
  in ten seconds and know what "the budget is exhausted" means for the
  next deploy.
- **The two burn-rate windows catch slow degradations.** A fast burn
  pages; a slow burn tickets. A single-window alert would miss the
  slow degradation that never triggers the fast threshold.
- **The absence of an SLA is documented.** A reviewer who asks "what
  happens if the service is down for a day" gets an honest answer (no
  SLA; the SLOs are internal) instead of a fabricated commitment.

**Negative / accepted trade-offs**

- **The SLOs are aspirational until M4.** The project has no production
  traffic, so the availability, latency, and quality numbers are targets
  the load test measures against. The ADR makes the aspirational status
  explicit; a reader who expects measured numbers is pointed to M4.
- **A solo project cannot enforce the freeze.** The policy is
  honor-system. The mechanism (alert + document) is what makes the
  discipline visible; a future team inherits the mechanism and can add
  enforcement.
- **The burn-rate constants are the SRE workbook's defaults.** They are
  config; the defaults are reasonable and documented. A deployment with
  a different traffic profile would tune them.
- **A 4xx spike is not an SLO breach.** A deployment whose client breaks
  suddenly sees no availability SLO change, which is correct (the
  service is fine) but means the SLO dashboard does not answer "is
  something wrong" for that class of incident. The 4xx metric's own
  alert is the answer.
- **The quality SLO's 1% threshold is a judgment.** A different traffic
  mix might set it at 0.1% (a hot cache) or 5% (a database with regular
  maintenance). The threshold is config; the reasoning is recorded.
- **A planned maintenance window is an exclusion.** A deployment that
  uses exclusions aggressively can hide failures from the SLO. The
  exclusion requires a declared window in the deployment manifest and
  an entry in the incident log; the paper trail is the guard.
- **No SLO for the pipeline or the churn endpoint.** The pipeline is
  covered by the eval gate; the churn endpoint (M7) can add its own
  SLO when it lands. The decision is a scope boundary, not an
  oversight.
- **The three SLOs are for one service.** A future microservice split
  (unlikely for this project) would need its own SLOs. The pattern here
  is one service's worth; the shape is documented for the split if it
  ever happens.

## References

- `docs/adr/0009-golden-set-and-metrics.md` — the offline metrics the
  SLOs are distinct from
- `docs/adr/0010-evaluation-thresholds.md` — the gate the SLOs are
  separate from
- `docs/adr/0014-rate-limiting.md` — why 429 is not an availability
  failure
- `docs/adr/0019-readiness-contract.md` — the readiness side of M6's
  rollback check
- `docs/adr/0020-fallback-chain.md` — the `meta.source` values the
  quality SLO reads
- `docs/adr/0021-observability-contract.md` — the metrics the SLOs are
  computed from
- `docs/runbook.md` § Service overview — the SLO table this ADR fills in
- `dashboards/slo.yaml` — the SLO definitions the dashboards read (M4)
- `docs/postmortems/` — the postmortem entries the error budget policy
  requires
