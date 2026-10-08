# ADR-0017: Experiment assignment and exposure logging

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

M5 ships the statistical machinery for A/B testing: SRM check, sample
size, significance, guardrails. None of that is meaningful without a
correct assignment at request time and an honest record of which variant
a user was actually served. A bug in either invalidates the experiment's
result, and a result that is not trustworthy is worse than no result.

The assignment rule is already fixed by `docs/adr/0009-golden-set-and-metrics.md`
§ 4: `sha256(f"{experiment_salt}:{user_id}") % 10_000`, mapped to
variants by traffic allocation, with a per-experiment salt. This ADR
covers what surrounds that rule:

1. **Where experiments are declared.** A committed file, a database, a
   feature-flag service?
2. **What salt is used and per what.** One salt per experiment, one
   per environment, or one for both?
3. **When an exposure is logged.** At assignment time, at response time,
   at the client's acknowledgment?
4. **What happens when an experiment is disabled, misconfigured, or
   being killed.** How does an operator stop an experiment without
   invalidating the data already collected?
5. **How the assignment is auditable.** What record does a reviewer have
   that the same user got the same variant for the same experiment?

These questions have to be answered before M3 writes any code that
serves a variant, because changing the answer later changes the meaning
of every experiment that ran under the old rule.

## Decision

### Experiments are declared in `experiments.yaml`, committed

The file lives at the repository root. It is read once at API startup
and cached in memory for the process's lifetime. It is not read from the
database, not from a feature-flag service, and not from an environment
variable. The reason is GitOps: an experiment is a change to behavior
that reviewers should see in a pull request, that history should record
in a commit, and that a running instance should be able to describe by
its own commit hash.

Schema:

```yaml
schema_version: 1
experiments:
  - name: rerank_mmr
    salt: "2026-10-08-rerank-mmr-v1"
    status: running           # running | paused | stopped
    allocation:
      control: 50
      treatment: 50
    started_at: "2026-10-08T00:00:00Z"
    stopped_at: null          # set when status becomes stopped
    description: >
      Compare the MMR-diversified re-ranker (treatment) against the
      blended-only re-ranker (control) on the same retrieval candidates.
```

Required fields: `name`, `salt`, `status`, `allocation`, `started_at`,
`description`. `stopped_at` is required when `status` is `stopped` and
must be null otherwise. Unknown fields are rejected.

**`status`** governs whether the experiment participates in assignment:

| Status | Behavior |
|---|---|
| `running` | Users are assigned and exposures are logged. |
| `paused` | No new assignments; requests that would have been assigned use the fallback variant (`control`) and log an exposure with `variant="control"` and `paused=true`. Used for a temporary hold without invalidating past data. |
| `stopped` | No assignment, no exposure logging, `control` served. `stopped_at` is set. Used when the experiment is finished or killed. |

`paused` and `stopped` differ in logging: `paused` continues to record
exposures (so the analysis can see the pause window), `stopped` does not
(the experiment is over). The distinction matters for a fixed-horizon
analysis (ADR-0009 § 6): a pause that lasts longer than planned should
not silently extend the experiment.

### One salt per experiment, namespaced by environment

Each experiment declares its own `salt`. Two experiments must not share
a salt: sharing would correlate their assignments (the same users would
land in the same bucket), which invalidates any comparison between them.

The salt in the file is **not** used directly. The assignment function
computes:

```
effective_salt = f"{environment}:{experiment.salt}"
```

where `environment` is `APP_ENV` (`dev` / `staging` / `prod`). The
environment prefix is prepended by the assignment code, not written into
the YAML. Two consequences:

- **A user gets a different variant in dev and prod for the same
  experiment.** This is desired: a developer testing the treatment in
  dev should not consume a prod bucket, and a user who happens to be
  both a developer and a customer is not confused by seeing the same
  variant in both.
- **Rotating environments does not require editing the YAML.** A
  promotion from staging to prod reuses the same `salt` string; the
  environment prefix makes the two assignments independent. A reviewer
  reading `experiments.yaml` sees the intent (`salt:
  "2026-10-08-rerank-mmr-v1"`), and the environment prefix is applied
  mechanically.

An operator who *wants* the same assignment across environments (a
deliberate override, e.g. to reproduce a user's experience in a
staging replay) can set the environment prefix to `prod` explicitly via
`EXPERIMENT_ENV_OVERRIDE`. This is not the default.

### Exposure is logged when the variant is served

Not when it is assigned. The two are different moments: a request assigns
a variant, then fails validation, then does not serve anything. An
exposure at assignment time would count that request as a trial of the
variant; the user never saw it, and the analysis would include a
phantom.

The exposure event is emitted on the response path, after all
middleware that can reject the request (auth, rate limit, validation)
and after the response is assembled. It carries:

- `experiment` — the name.
- `variant` — the variant the user was served.
- `request_id` — the request's correlation id.
- `user_id_hash` — the HMAC-hashed user id (ADR-0018), never the raw id.
- `assignment` — a debug field: the bucket number (`0..9999`) and the
  effective salt's version tag. Present so a reviewer can reproduce the
  assignment from the log without a second lookup.

Exposure logging is **synchronous**. The event goes to the same event
store as `POST /v1/events` (a PostgreSQL insert). If the insert fails,
the failure is logged and counted (`recsys_experiment_exposure_errors_total`)
but the response is still returned. The alternative — an async queue —
adds a component the project does not have at M3; a synchronous insert
into the same store the events endpoint writes to is one less moving
part.

### Exposure is idempotent by `request_id`

The event store's `event_id` uniqueness (ADR-0018) applies to exposures
too: the exposure's `event_id` is derived from `request_id` and the
experiment name (`sha256(f"exposure:{experiment}:{request_id}")`).

The reason is retries. A client that does not receive a response (a
timeout, a network drop) may retry the same request. Without
idempotency, the same user is counted twice in the experiment. With it,
the duplicate is counted and ignored. A client that legitimately makes
two requests gets two exposures (two different `request_id`s), which is
correct — the user saw the recommendation twice.

### Fallback variant is `control`

Three cases produce a fallback:

1. **Experiment is `paused` or `stopped`.** The user is served `control`.
2. **`EXPERIMENT_DISABLED=1` is set.** Every experiment is treated as
   `stopped` for this process, and `control` is served. This is the kill
   switch; it is an environment variable, not a YAML edit, so the
   operator can disable an experiment without a deploy. The variable is
   read at startup; changing it requires a restart, which is the right
   trade-off (a mid-process flip of a running experiment is a worse
   surprise than a restart).
3. **Allocation does not cover the user's bucket.** If the sum of
   allocations is less than 100 (a misconfiguration), buckets above the
   sum fall through to `control`. This is a safety net, not a
   configuration option: a well-formed `experiments.yaml` sums to 100.

**A fallback assignment still logs an exposure**, with `variant="control"`
and `fallback_reason` set. Skipping the exposure would leave the
analysis unable to distinguish "user was in control" from "user was
supposed to be in an arm that no longer exists". The event is data; the
reason is the metadata that makes the data interpretable.

### Validated at startup, fails fast

`experiments.yaml` is validated when the process starts. A malformed
file aborts startup with a clear error. The checks:

- `schema_version == 1`.
- `name` is a non-empty string, unique within the file.
- `salt` is a non-empty string, unique within the file. Two experiments
  with the same salt is the most likely way to silently correlate two
  experiments; the check makes it a startup error.
- `status` is one of `running`, `paused`, `stopped`.
- `allocation` keys are non-empty strings; values are integers in
  `[0, 100]`; the sum is exactly `100`. A sum other than 100 is a
  misconfiguration and fails at startup rather than at the first
  request that hits the uncovered bucket.
- `started_at` parses as ISO 8601 UTC.
- `stopped_at` is null when `status != "stopped"`, and a valid ISO 8601
  timestamp when `status == "stopped"`.

Failing at startup, rather than degrading to `control` for a malformed
file, means a typo in the YAML is caught before it produces traffic with
an unintended assignment. The fallback in the previous section is for
*paused* or *stopped* experiments, not for malformed ones.

### Not in M3

Three things are deliberately out of scope, recorded here so their
absence is a decision rather than an oversight:

- **Database-backed allocation.** A DB-backed allocation would let an
  operator change a running experiment's traffic split without a
  deploy. It is also a stateful component the project does not have, and
  its failure mode (an experiment that cannot be read while the DB is
  down) is worse than a deploy's. M5 or later can add it if the deploy
  cadence proves painful.
- **A feature-flag service (Unleash, Flagsmith).** Same reasoning as
  above, plus a second external dependency. Not needed at this scale.
- **Per-user overrides (QA allowlist).** A QA engineer who wants to see
  the treatment can run a local instance with `EXPERIMENT_ENV_OVERRIDE`
  set to a value that lands them in the treatment bucket; a per-user
  override is a code path that exists only to serve a workflow the
  override already supports.

## Alternatives considered

| Option | Why not |
|---|---|
| **Experiments declared in the database** | Requires a migration and a query on every request (or a cache with its own invalidation). The GitOps flow (PR review, commit history, `git blame`) is worth more than the runtime flexibility, which the `paused` status already provides for the common case. |
| **A feature-flag service (Unleash, Flagsmith)** | Adds a second external dependency and a second set of semantics (its assignment model, its UI, its SLA). Correct for a multi-team product; wrong for a single-operator project with one experiment at a time. |
| **One salt for all experiments** | Correlates assignments: the same users land in the same bucket of every experiment, so two experiments are not independent. Any comparison between them is confounded. |
| **The salt includes the environment in the YAML** | Requires editing the YAML for every promotion, and any drift between the dev file and the prod file is invisible until someone reads both. The environment prefix in code is mechanical and cannot drift. |
| **Log the exposure at assignment time** | Counts requests that were assigned but never served (validation error, rate limit, ANN failure). The analysis would include phantom trials. |
| **Log the exposure from the client (an explicit event)** | The client may not send it, may delay it, or may send it multiple times. The server knows when it served a response; that is the authoritative moment. |
| **Async exposure logging via a queue** | Adds a component (the queue) and a failure mode (the queue is full) for a write that is small and synchronous on the same store the events endpoint uses. A future ADR adds async when the write is measured to be on the hot path; not now. |
| **No idempotency on the exposure** | A retry (network timeout) counts the same user twice. The `request_id`-derived id is free to add and removes a whole class of double-counting. |
| **Kill switch as a YAML edit** | Requires a deploy to flip. The whole point of a kill switch is a fast, non-deploy stop. `EXPERIMENT_DISABLED` as an env var meets that; the YAML status covers the "planned stop" case. |
| **Kill switch as a runtime-mutable flag (Redis, SIGHUP)** | A mid-process flip of a running experiment is a surprise the operator did not ask for; the restart cost is small and the semantics are simpler. |
| **Sum of allocations allowed to be `< 100`** | The uncovered buckets have no defined behavior. Failing at startup makes the misconfiguration visible; the fallback for buckets above the sum exists only as a safety net for a case the validation already catches. |
| **A separate "evaluation assignment" for offline runs** | The offline evaluation (ADR-0009) does not use experiment assignment; it runs the retrieval path directly. Introducing an assignment there would produce offline metrics for a variant that production might not be serving. |
| **Assignment computed at analysis time from `user_id_hash`** | The salt can change (a new experiment reuses the name), the environment can differ, and the analysis should not reimplement the assignment. Recording the bucket at serve time is the audit trail. |

## Consequences

**Positive**

- **Assignment is auditable from the commit hash alone.** The file and
  the code that reads it are both in the repository; a reviewer can
  reproduce any user's variant from the file at the commit that served
  them.
- **The environment prefix prevents cross-environment contamination.**
  A developer in dev does not consume a prod bucket; a user who
  happens to have an account in both gets independent assignments.
- **Exposure reflects what was served.** A failed request does not
  count; a successful response does. The M5 analysis starts from an
  honest record.
- **The kill switch does not require a deploy.** An operator who sees a
  guardrail breach (M5) can stop the experiment in seconds.
- **A misconfigured `experiments.yaml` fails at startup.** The failure
  mode is "the process does not start", not "the process serves traffic
  with an unintended assignment".

**Negative / accepted trade-offs**

- **Adding or changing an experiment requires a deploy.** A traffic
  split cannot be adjusted in real time. Accepted at this scale; a
  DB-backed allocation is a future ADR if it becomes painful.
- **The fallback variant is always `control`.** There is no way to send
  a paused experiment's traffic to `treatment`, which is correct (the
  point of pausing is to stop the treatment) but is a behavior a future
  experiment might want to change. If a use case appears, an ADR adds a
  `paused_variant` field.
- **`EXPERIMENT_DISABLED` reads at startup.** A flip requires a
  restart. This is deliberate (see Context) but means an operator with a
  hot-reload workflow would be surprised. The startup log line prints
  the variable's value so the state is visible.
- **The exposure is synchronous.** A slow database insert adds latency
  to the response. The insert is a single row with a primary-key
  conflict check; on the target PostgreSQL it is a few milliseconds.
  M4's load test measures the actual cost.
- **The `assignment` debug field leaks the bucket number.** A caller who
  can read their own bucket number can infer the assignment of any user
  whose `user_id_hash` they know (by hashing with the same salt and
  checking the bucket range). This is not a security issue — the
  assignment is not a secret and the analysis is aggregate — but it is
  a property worth stating. A future ADR could remove the field if a
  use case requires it.
- **The schema validation runs once at startup.** A malformed file
  edited after startup (a manual change on a running instance) is not
  noticed until restart. The file is in git and the running instance's
  commit is in the response's `meta`; the drift is visible, not silent.
- **The `salt` string is operator-chosen.** A salt that is reused across
  two logically different experiments (a rename) silently correlates
  them. The startup check catches duplicates within one file, but the
  file has no memory of past experiments; the convention is a salt
  string that includes a version suffix (`...-v1`, `...-v2`), documented
  in `docs/ops.md`.

## References

- `docs/adr/0009-golden-set-and-metrics.md` § 4 — the assignment rule
  this ADR surrounds
- `docs/adr/0018-event-ingestion.md` — the event store exposures are
  written to, and the `user_id_hash` they carry
- `docs/contracts.md` § 1.1 — the `ExperimentRef` shape in an event
- `docs/ops.md` — the salt convention and the kill-switch procedure
- `experiments.yaml` — the committed experiment declarations
- `src/recsys/experiments/assignment.py` — the assignment function
- `src/recsys/experiments/exposure.py` — the exposure writer
- `tests/unit/test_assignment.py` and
  `tests/integration/test_exposure_logging.py`
