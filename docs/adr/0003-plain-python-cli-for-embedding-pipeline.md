# ADR-0003: Plain Python CLI for the embedding pipeline

- **Status:** Accepted
- **Date:** 2026-10-07
- **Deciders:** project maintainer

## Context

M1 must produce deterministic embeddings from a catalog snapshot, verified by
a checksum contract (see `docs/embedding-pipeline.md` § 5). The pipeline has
to run in three places:

- The primary development environment (Google Colab), which has no scheduler
  and no Docker daemon.
- CI, where it is exercised by the integration job.
- The eventual production deployment, where the same pipeline is expected to
  run on a schedule once M6 lands.

`README.md` § Tech stack originally named "Prefect or Airflow" as the batch
pipeline. That choice was made at design time, before M1 scoped what the
pipeline actually needs to do. The question this ADR settles is whether
orchestration belongs in M1 or in a later milestone.

## Decision

M1 ships the pipeline as a **plain Python CLI**: `python -m
recsys.embeddings.pipeline --mode={batch,incremental}`. It reads a catalog,
writes versioned artifacts, and exits with a non-zero status on failure.

Orchestration (Prefect, Airflow, or a cron wrapper) is deferred to **M6**,
where scheduling, retries, and alerting become real requirements. Whatever
framework is chosen then will **wrap** this CLI, not replace it. The CLI's
arguments, exit codes, and JSON stdout summary are the integration contract.

## Alternatives considered

| Option | Why not |
|---|---|
| **Prefect in M1** | Adds a server or cloud account, a UI to learn, and a concept graph (flows, tasks, deployments) that M1 does not use. Retry, schedule, and alerting — Prefect's actual value — are not exercised until M6. Shipping the framework without the requirements it addresses is cargo-culting. |
| **Airflow in M1** | Heavier than Prefect: scheduler, metastore database, web server. Same mismatch as above, with more infrastructure. |
| **Dagster in M1** | Same category as Prefect; its asset-graph model is a good fit conceptually but is not justified by M1's scope. Reconsidered in M6. |
| **Makefile-only, no Python entry point** | A `make embed` recipe is convenient but does not give us a callable program with arguments, exit codes, and structured output. CI, tests, and M6 orchestration all need a real entry point. The Makefile target becomes a thin wrapper around the CLI. |
| **Library-only (import and call from a notebook or script)** | Works for exploration but not for reproduction: whoever runs it later has to know which function to call, with which arguments. A CLI is self-documenting via `--help` and produces a stable JSON summary line for automation. |

## Consequences

**Positive**

- The pipeline is debuggable in any environment that runs Python: Colab,
  CI, a laptop, or a container. No scheduler to install, no server to reach.
- Exit codes and the JSON stdout summary form a stable integration surface.
  Tests, CI, and any future orchestrator consume the same interface.
- M6's choice of orchestrator is genuinely open. The requirements that drive
  the choice (retry policy, backfill, alerting, dependency graph) are
  evaluated at the moment they exist, not guessed two milestones earlier.
- The CLI is the smallest possible surface for the determinism contract in
  `docs/embedding-pipeline.md` § 5. Fewer moving parts means fewer ways for
  the checksum to drift between runs.

**Negative / accepted trade-offs**

- No retry, no backoff, no scheduling out of the box. In M1 this is fine:
  the pipeline is invoked manually (`make embed`) or by CI. The gap becomes
  visible when the pipeline needs to run unattended — that gap is the M6
  trigger.
- No run history or UI. Failures are visible only through CI logs, exit
  codes, and the JSON summary line. For a single-operator project this is
  adequate; if a second operator joins before M6, the gap should be revisited.
- The eventual orchestrator will have to wrap the CLI rather than drive the
  pipeline through its native API. This is deliberate: the CLI is the stable
  boundary, and the orchestrator is replaceable. A future ADR may revisit
  this if a specific orchestrator requires deeper integration than
  subprocess execution.

## References

- `docs/embedding-pipeline.md` — the pipeline design this ADR scopes
- `docs/decisions.md` § 5 (ingestion modes) — updated by this ADR
- `docs/adr/0002-onnx-runtime-for-inference.md` — the encoder used by the
  pipeline
- M6 milestone (`README.md`) — where orchestration requirements land
