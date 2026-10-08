# ADR-0023: Deployment strategy

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

M6 ships automated deployment: a merge to `main` reaches staging, then
promotes to production after approval, with automatic rollback when
readiness or SLO checks fail. The design has to answer six questions
before that milestone:

1. **What is deployed?** A container image tagged with the git SHA is
   the obvious unit. But three artifacts change on different cadences:
   the API image, the embedding runs (`artifacts/embeddings/`), and the
   active index (`index_registry`). Rolling them together couples the
   deploy of a code change to the deploy of a data change, which is a
   mistake.
2. **How does a new API image become live without dropping requests?**
   A rolling update with readiness probes (ADR-0019) is the pattern.
   The details — how many instances, how long between steps, what
   triggers a stop — matter.
3. **How does a schema migration interact with a rolling update?** A
   migration that adds a column is safe mid-roll; a migration that
   renames a column is not. The rule has to be stated.
4. **What is the rollback story for code, embeddings, and indexes?**
   Three separate operations with different timescales.
5. **What is a canary, and does this project need one?**
6. **What does the deployment do when a step fails?** Fail the whole
   pipeline, or fail just the step and continue?

## Decision

### Three artifacts, three lifecycles, three pointers

| Artifact | Pointer | Change cadence | Deploy mechanism |
|---|---|---|---|
| **API image** | The running container's tag | Per merged PR | Rolling update (below) |
| **Embedding run** | `artifacts/embeddings/current` (ADR-0004) | Per catalog or model change | The pipeline writes the pointer; consumers read on demand |
| **Active index** | `index_registry.status = 'active'` (ADR-0006) | Per index build (after an embedding run) | `promote_index.py` updates the row; the API polls (ADR-0022) |

**They are decoupled.** A deploy of a new API image does not rebuild an
index; a new index does not require a new image. The reasons:

- The index's identity (ADR-0007) depends on the model version, the
  catalog snapshot, the HNSW parameters, and the pgvector version.
  None of those changes when the API code changes.
- An API change that would affect retrieval (a new filter field, a new
  re-ranker weight) can be deployed and tested against the existing
  index; a separate promote is the response if the change requires one.
- Coupling them would mean that every code change pays the cost of an
  index rebuild (minutes on a large catalog) or that every index rebuild
  requires a code deploy.

The three pointers are the interfaces. A deploy that needs a new index
does both, in order: the embedding run (if the model changed), the
index build, the promote, and the API rollout. A deploy that does not
need a new index does the API rollout alone.

### API rollout: rolling update, one instance at a time

The container orchestrator (Kubernetes in M6, or a `docker compose`
equivalent in a smaller deployment) does a rolling update with:

- **`maxSurge: 1`, `maxUnavailable: 0`.** A new instance starts before
  an old one stops; at no point is the instance count below the desired
  count.
- **Readiness probe** on `/readyz` (ADR-0019). The orchestrator sends
  traffic to a new instance only after `/readyz` returns `ready`.
- **`terminationGracePeriodSeconds`** covering the longest in-flight
  request plus a margin. Default 30 seconds; the API's slowest expected
  request is bounded by the ANN timeout plus the fallback budget, well
  under that.
- **`preStop` hook** that sleeps `PRE_STOP_DELAY_SECONDS` (default 5)
  before the process receives `SIGTERM`. The sleep exists because the
  load balancer's removal of the instance from rotation is asynchronous
  with the orchestrator's view; a small delay prevents a request that
  the balancer already routed to an instance that is about to stop.

**Rollback is a new rollout of the previous image.** The previous tag
is the previous commit's SHA, and it is already in the registry; the
rollback deploys it and the readiness probes confirm it is healthy.
There is no "undo migration" in the rollback path (below).

### Canary: a manual step, not an automatic one

An automatic canary — 5% of traffic for N minutes, promote if healthy —
is not part of M6. The reasons:

- **Traffic-shaping infrastructure.** A canary needs a mechanism to
  route a fraction of traffic to the new image (a service mesh, a
  weighted load balancer, or an ingress with weighted backends). The
  project's single-node target does not have one, and adding one for
  the canary is disproportionate.
- **Sample size.** At the target RPS (M4), 5% of traffic for 5 minutes
  is a small sample against the p95 target; a canary that cannot
  distinguish a regression from noise is a canary that promotes on
  noise.
- **The rolling update is already a canary.** The first new instance
  receives a fraction of traffic (1/N of the deployment) until its
  readiness passes and the next instance starts. The window between
  the first instance's readiness and the last instance's replacement is
  the canary window. An operator watching the rollout sees the new
  version's behavior on a subset of traffic before the rollout
  completes.

What M6 does instead: **a manual pause between staging and production.**
The pipeline deploys to staging automatically, runs the smoke tests
and the evaluation gate (M2), and waits for a human approval to
promote to production. The staging environment is a real canary in the
sense that it runs the same code against a real database; the human
approval is the decision point.

An automatic canary is a candidate for a future ADR if the project
grows a deployment that can shape traffic and a traffic level that
makes 5% a meaningful sample.

### Migrations: additive only during a rollout

A rolling update has old and new code running at the same time for the
duration of the rollout. A migration that changes the schema in a way
the old code does not understand will break the old instances before
they are replaced.

The rule, applied to every migration:

- **Additive changes are safe mid-roll.** Adding a nullable column,
  adding a table, adding an index.
- **Destructive changes are not safe mid-roll.** Dropping a column,
  renaming a column, changing a column's type in a way that loses data,
  adding a `NOT NULL` constraint to an existing column with rows.
- **A destructive change is two deploys.** First deploy: add the new
  shape and write to both (the old code reads the old shape, the new
  code writes both). Second deploy, after every instance runs the new
  code: drop the old shape.

The rule is documented in `docs/ops.md` § Migrations and is the
responsibility of the migration's author, not of the CD pipeline. The
pipeline runs `alembic upgrade head` before the new instances start;
the rule is what makes that step safe.

**A migration that cannot be made additive is a manual operation.** A
column rename with data, a type change that loses precision, a table
merge — these run in a maintenance window with the API drained. The
runbook for a manual migration is in `docs/ops.md`; the CD pipeline
does not attempt them.

### Deployment order for a change that touches more than one artifact

Some changes need a new embedding run or a new index. The order:

1. **Embedding run.** The pipeline writes a new run and updates
   `artifacts/embeddings/current` (ADR-0004). Consumers read the new
   run on demand.
2. **Index build.** `build_index.py` reads the active run and writes a
   `building` row in `index_registry` (ADR-0006).
3. **Index promote.** `promote_index.py` promotes the `building` row
   and retires the previous active one. The API's poll (ADR-0022)
   picks up the new active version within 10 seconds.
4. **API rollout** (only if the API code changed).
5. **Cache warmup** (best-effort, after step 3; ADR-0015).

**Steps 1–3 are separate from 4.** A pipeline that bundles them into a
single job makes the failure modes harder to distinguish and prevents
re-running one step without re-running the others. The CD workflow in
M6 orchestrates them as separate steps; a failure in step 2 does not
re-run step 1.

**Step 4's rollout is a rolling update, not a blue/green swap.** For
the API image, a rolling update with readiness probes is the standard
pattern; a blue/green swap (two full fleets, a single traffic switch)
is more infrastructure than the project has. The index's blue/green
swap is a different mechanism (a pointer in a database row, ADR-0006)
and is not the same as a blue/green deployment of the API.

### What "automatic rollback" means

The CD pipeline's automatic rollback (M6) covers one case: **a rollout
whose new instances never become ready**. The readiness probe fails on
the new image, the orchestrator stops the rollout, and the pipeline
deploys the previous tag.

The cases the automatic rollback does **not** cover:

- **A code regression that passes readiness.** A new image that serves
  500s on a specific input is ready (readiness checks dependencies, not
  every route) and the rollout completes. The rollback for this case is
  a human decision, informed by the metrics (ADR-0021) and the alerts.
- **A performance regression.** The new image serves correctly but is
  slower. The p95 alert (M4) fires; the human decides whether to roll
  back or to accept the change.
- **A schema migration that is already applied.** A code rollback does
  not undo a migration. The migration's additive-only rule (above)
  means the old code runs against the new schema; if the migration was
  destructive, the rollback is not possible without a manual restore.

**The automatic rollback is narrow on purpose.** A pipeline that
automatically rolls back on a metric it did not measure (a p95 it
cannot see) would be rolling back on inference. The narrow rollback —
readiness failure — is a signal the pipeline has, and it is the signal
that means "the new image cannot serve at all".

### Failure semantics: stop the pipeline, do not continue

A failed step stops the pipeline. The reasons:

- **A smoke test failure on staging means the new image does not
  work.** Promoting it to production would deploy a known-bad image.
- **An evaluation gate failure (M2) means the code changed retrieval
  quality.** Promoting it would deploy a regression the gate exists to
  catch.
- **A readiness failure in production means the new image cannot
  serve.** Continuing to the next step (cache warmup, metric check)
  would work against a system that is not serving.

The pipeline's steps are ordered and each depends on the previous. A
failure at step N stops the pipeline and leaves the system in step
N-1's state, which is the last known-good state (or a partially
rolled-out state that the orchestrator's rollback completes).

**The one exception: cache warmup.** A failed warmup after a
successful promote is logged and the pipeline continues. Warmup is
best-effort (ADR-0015); a failure means the cache is cold, not that the
deploy failed.

### Staging is a real environment, not a mock

The staging environment (M6) is a full deployment: a database, a Redis,
an API, a real index. It is not a test harness. The reasons:

- **The evaluation gate (M2) needs a real index.** A mock would not
  exercise the retrieval path the gate measures.
- **The smoke tests need real dependencies.** A test against a mock
  proves the mock works.
- **A bug that only appears with a real database is a bug the staging
  environment exists to find.**

Staging's data is a copy of the sample catalog (for the project's
scale), not production data. The environment is destroyed and recreated
per deploy, so a failed staging deploy leaves nothing to clean up.

### What is not in M6

- **Multi-region deployment.** Out of scope (README § What this is
  not).
- **Blue/green for the API image.** Rolling update is enough; a
  blue/green fleet doubles the resource cost for a faster cutover the
  project does not need.
- **Automatic canary with traffic shaping.** See above.
- **Automated database migrations with a data backfill.** The
  additive-only rule covers the common case; a backfill is a script
  the operator runs, documented in the runbook.
- **Chaos testing.** Not this project's scope.

## Alternatives considered

| Option | Why not |
|---|---|
| **Blue/green for the API image** | Doubles the resource cost for a cutover that a rolling update already makes smooth. The rolling update's readiness probe is the mechanism; a second fleet is not needed at this scale. |
| **Recreate (stop all, start all)** | Drops every in-flight request. The rolling update is the standard alternative and the project's target scale does not make it expensive. |
| **Automatic canary with a weighted load balancer** | Adds a mechanism (weighted routing) and a sample-size problem (5% for 5 minutes is too few requests to distinguish a regression). The rolling update is the canary; the human approval between staging and production is the decision. |
| **Deploy the API, the embedding run, and the index as one job** | Couples three artifacts with different cadences. A code deploy would pay an index rebuild's cost; a model upgrade would pay a code deploy's risk. Separate steps, separate failures. |
| **A single `current` pointer for all three artifacts** | Same coupling, one pointer. The three pointers exist because the three artifacts change independently. |
| **Automatic rollback on every alert** | The pipeline does not measure p95 or error rate (M4's dashboards do). Rolling back on inference is not automatic; it is a guess. The narrow rollback (readiness) is the signal the pipeline has. |
| **Continue the pipeline after a smoke test failure** | Deploys a known-bad image. A failure stops the pipeline; the human decides. |
| **Roll back the migration on code rollback** | Alembic's `downgrade` is not always safe (it may lose data), and the old code may not understand the downgraded schema if the downgrade is incomplete. The additive-only rule is what makes a code rollback safe against a new schema. |
| **Skip staging for a "small" change** | The definition of "small" is the thing that is wrong the first time a small change breaks production. Every merge to `main` goes through staging. |
| **A separate staging database that is not a copy** | A staging database with different data tests the schema, not the queries. The sample catalog is the data; the schema is the same. |
| **A canary that samples by user id (a stable bucket)** | Would compare the same user's behavior across versions, which is a good design for A/B tests but not for a deploy canary (the deploy canary's question is "does the new code work", not "does the new code help"). The rolling update's natural fraction is the sample. |
| **No preStop delay** | The load balancer's view and the orchestrator's view are asynchronous; without a small delay, a request can be routed to an instance that has just been told to stop. Five seconds is cheap. |
| **`maxUnavailable: 1`** | Allows the fleet to be one instance below the desired count during the rollout. On a two-instance deployment that is a 50% capacity drop. `maxSurge: 1, maxUnavailable: 0` keeps the count at or above the desired count throughout. |

## Consequences

**Positive**

- **A code deploy does not pay for an index rebuild.** The three
  artifacts are decoupled; a change that only touches the API is a
  container rollout.
- **A rolling update does not drop requests.** `maxSurge: 1` starts a
  new instance before stopping an old one; readiness gates the traffic
  shift; the preStop delay covers the balancer's async view.
- **The migration rule is one sentence.** Additive changes are safe
  mid-roll; anything else is two deploys or a manual operation. A
  reviewer of a migration has one question to ask.
- **The automatic rollback is narrow and honest.** It fires on the one
  signal the pipeline measures (readiness) and does not pretend to
  measure what it does not.
- **The canary is the rolling update itself.** No traffic-shaping
  infrastructure, no sample-size problem; the first instance's traffic
  is the fraction.

**Negative / accepted trade-offs**

- **A code regression that passes readiness is not caught
  automatically.** The pipeline rolls out a code change that serves
  correctly for readiness and incorrectly for a specific input. The
  metric alert (M4) fires and a human responds. The alternative — a
  pipeline that measures every route's behavior — is a canary with
  traffic shaping the project does not have.
- **A schema migration that is not additive requires a manual
  operation.** The pipeline does not attempt it. A deployment that
  needs one pauses and an operator runs the runbook. The cost is a
  slower deploy for a change the rule discourages.
- **Three pointers to keep in sync.** A deploy that changes all three
  has three steps; a step skipped is a mismatched state (an API
  expecting a filter field the index does not have, a new index for an
  old model). The CD pipeline's step ordering (above) is the discipline.
- **Staging's data is not production's data.** A bug that depends on
  production data's distribution does not appear in staging. The
  sample catalog is what the project has; a larger catalog is a future
  concern.
- **No automatic canary means a human is in the loop for every
  production deploy.** For a portfolio project, that is the intended
  behavior (the human approval is the decision point). For a larger
  team, an automatic canary would be worth the infrastructure; the ADR
  records the door as open.
- **The preStop delay adds 5 seconds to every roll.** On a deploy that
  rolls 10 instances sequentially, 50 seconds of rollout time. The
  alternative (a request dropped during the balancer's async view) is
  worse.
- **A migration that a reviewer fails to spot as non-additive ships.**
  The rule is documented; the review is the enforcement. A linter for
  "this Alembic revision contains a DROP" is a possible future
  addition; M6 relies on the review.

## References

- `docs/adr/0004-versioned-runs-with-current-pointer.md` — the
  embedding-run pointer
- `docs/adr/0006-pgvector-schema.md` — the `index_registry` row and the
  promote procedure
- `docs/adr/0007-index-version-identity.md` — why a code change does
  not change the index version
- `docs/adr/0015-cache-and-circuit-breaker.md` — cache warmup after
  promote
- `docs/adr/0019-readiness-contract.md` — the readiness probe the
  rolling update gates on
- `docs/adr/0022-config-management.md` — the poll that picks up the new
  active index
- `docs/ops.md` § Migrations — the additive-only rule and the manual
  migration runbook
- `docs/runbook.md` — the rollback procedures
- `.github/workflows/cd.yml` — the pipeline (M6)
