# ADR-0022: Configuration management

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

By M3 the project reads configuration from four places with different
lifetimes and different sensitivities:

1. **Environment variables.** `Settings` (Pydantic Settings) reads
   `DATABASE_URL`, `REDIS_URL`, `API_KEYS`, `JWT_SECRET`, the HNSW
   parameters, the timeout values, and so on. `docs/contracts.md` § 3 is
   the contract.
2. **A committed YAML file.** `experiments.yaml` (ADR-0017) declares
   experiments and their salts. It is read at startup and does not
   change while the process runs.
3. **A committed YAML file.** `evaluation/thresholds.yaml` (ADR-0010) is
   read by `make eval`, not by the API. The API does not see it.
4. **Database rows.** `index_registry` (ADR-0006) carries the active
   index's version and parameters. The API reads it at startup and
   caches it; a change requires a signal (an endpoint, a restart, or a
   poll) for the running instance to notice.

Four sources with three different lifetimes (process, file, row) and
two different sensitivities (secret, non-secret) create a class of
questions M3 has to answer before writing more code:

- **Precedence.** If `RATE_LIMIT_PER_MINUTE` is set in the environment
  and `experiments.yaml` references a limit, which wins?
- **Reload.** Which config changes require a restart, and which can be
  picked up while the process runs?
- **Secrets.** What is a secret, where may it live, and how is it
  prevented from appearing in a log line or a metric label (ADR-0021)?
- **Validation.** When is a misconfiguration caught — at startup, at
  first use, or never?
- **Test isolation.** How does a test set a config value without
  touching the real environment or the file on disk?

The M0 scaffold uses Pydantic Settings with a `Settings` singleton
(`get_settings`, cached with `lru_cache`). It validates some fields
(HNSW parameters, cache TTL) and refuses to boot in prod with dev
defaults for secrets. This ADR extends that foundation with the rules
the M3 surface needs.

## Decision

### Five categories, each with one source of truth

| Category | Source | Example | Restart to change? |
|---|---|---|---|
| **Boot config** | Environment variables via `Settings` | `DATABASE_URL`, `JWT_SECRET`, `API_KEYS`, `HNSW_M` | Yes |
| **Hot config** | A watched file (see below) | rate limits, re-ranker weights | No |
| **Static domain data** | Committed YAML | `experiments.yaml` | Yes |
| **Runtime state** | A database row | the active `index_version` | No (signal) |
| **Evaluation config** | Committed YAML | `evaluation/thresholds.yaml` | Not read by the API |

**Every value belongs to exactly one category.** A value that appears in
two categories is a bug, and the code that reads it has to say which one
it is reading.

### Boot config: environment variables, read once, validated at startup

`Settings` is instantiated once at process start, cached by
`get_settings`, and never re-read. A change to an environment variable
requires a restart. The reasons:

- **Reproducibility.** The process's behavior is a function of the
  environment it started in, which is what an operator reads from the
  deployment manifest.
- **Validation cost.** Pydantic validates each field once; a
  per-request re-read would validate on every request or cache the
  validation with its own invalidation problem.
- **Secrets.** An environment variable that is re-read invites code
  paths that re-read it in places where a logging bug could capture it.
  One read at startup means one moment when a secret is in memory as a
  string, and it is the moment the settings object is constructed.

**Validation is fail-fast.** `Settings` validates every field's type
and range at construction. A missing `DATABASE_URL` in prod, an invalid
`INDEX_BACKEND` value, an HNSW configuration that violates
`ef_construction >= m`, a sentinel `JWT_SECRET` in prod — all raise
before the app serves a request. The M0 prod guard is extended:

| Field | Prod rule |
|---|---|
| `API_KEYS` | Non-empty |
| `API_KEYS_ADMIN` | May be empty (a deployment with no admin operations is legal) |
| `JWT_SECRET` | Not the sentinel value from `.env.example`; at least 32 characters |
| `USER_ID_HASH_SALT` | Set, non-empty, and `USER_ID_HASH_SALT_VERSION > 0` |
| `DATABASE_URL` | Present and parseable as a URL |
| `REDIS_URL` | Present and parseable as a URL |

The check runs in `Settings._prod_guard`, so a deployment that
misconfigures any of these fails at container start, not on the first
request that needs the value.

### Hot config: a watched YAML file

Three values change frequently enough that a restart is friction, but
not so frequently that they need a database:

- `RATE_LIMIT_RECOMMEND_PER_MINUTE` and the other rate-limit classes
  (ADR-0014).
- The re-ranker weights (`RERANK_W_SIM`, `RERANK_W_POP`, `RERANK_W_REC`)
  and `RERANK_MMR_LAMBDA` (ADR-0016).
- The cache TTLs (`CACHE_TTL_RECOMMEND_SECONDS`,
  `CACHE_TTL_SIMILAR_SECONDS`).

These live in a single committed file, `config/hot.yaml`:

```yaml
schema_version: 1
rate_limit:
  recommend_per_minute: 600
  events_per_minute: 6000
  admin_per_minute: 60
rerank:
  w_sim: 0.7
  w_pop: 0.2
  w_rec: 0.1
  mmr_lambda: 0.7
cache:
  recommend_ttl_seconds: 300
  similar_ttl_seconds: 600
```

**The file is read at startup and re-read on every change.** The
watcher is a background thread that polls the file's `mtime` every
`HOT_CONFIG_POLL_SECONDS` (config, default 5) and, on a change, loads
the new document, validates it, and swaps the in-memory hot config.

**Polling, not `inotify`.** `inotify` is Linux-only and behaves
differently on different filesystems (some containers have a stale
`inotify` after a bind-mount update). A 5-second poll is one `stat`
call, cheap, portable, and it catches a change within a window that is
acceptable for the values it controls. A change to a rate limit that
takes effect 5 seconds after the file is saved is not a problem.

**A malformed hot config does not abort the swap.** The validation
runs before the swap. A file with an invalid value (a negative weight,
a rate limit of zero, a TTL above the schema's maximum) is logged at
ERROR with the offending field and the previous value stays in effect.
The process does not crash and does not silently accept the bad value.

**Why these three and not more.** The criterion is: *does the value
need to change without a deploy, and does a wrong value have a bounded
blast radius?*

- A rate limit changed mid-incident is a common operational move; a
  wrong value throttles too much or too little and is correctable by a
  second edit.
- A re-ranker weight changed to compare two configurations is the
  experiment-arm workflow (ADR-0017) in miniature; a wrong value changes
  ranking quality, not correctness.
- A cache TTL changed during an incident (longer for a hot period) is
  a common move; a wrong value bounds staleness, which is already
  bounded by the index's update cadence.

A value that fails the criterion stays a boot config. `DATABASE_URL`,
`JWT_SECRET`, and `HNSW_M` all require a restart because their wrong
values have unbounded blast radius (a database the app should not talk
to, a secret that invalidates every token, an index whose parameters no
longer match the built index).

**Hot config does not include secrets.** The file is committed. A
secret in it would be a secret in git, which is the failure mode the
`.env.example` and `.gitignore` rules exist to prevent. The validation
rejects a hot config that contains a field named `secret`, `password`,
`token`, `key`, or `salt`.

### Static domain data: `experiments.yaml`, read once

`experiments.yaml` (ADR-0017) is committed, read at startup, and
validated. Changing it requires a restart. The reasons are already in
ADR-0017 (GitOps, auditability); this ADR adds the placement: it is a
*static* category, not a *hot* one, because a mid-process change to an
experiment's allocation would make the exposure log ambiguous (which
allocation was in effect at each exposure?). A deploy is the boundary at
which the assignment changes, and that is deliberate.

### Runtime state: the active index, read at startup, refreshed on signal

The active `index_version` lives in `index_registry` (ADR-0006). The API
reads it at startup and caches it. A change (a promote or a rollback,
ADR-0006 § Promote) takes effect on a running instance only when the
instance notices.

**M3 uses a signal, not a poll.** The promote and rollback scripts
(ADR-0006) update the registry row; they do not signal the API directly
because the API may have several instances and the scripts do not know
their addresses. M3 introduces a lightweight in-process cache with a
short TTL: the API re-reads the registry row at most once every
`ACTIVE_INDEX_POLL_SECONDS` (config, default 10) on the request path.

Ten seconds is a trade-off. A promote takes effect within ten seconds on
every instance, without a service-discovery mechanism and without the
promote script having to know the API's addresses. The cost is one
`SELECT` per instance per ten seconds, which at ten instances is one
query per second against a small indexed table. If the query becomes a
concern, a longer TTL or an explicit signal (an authenticated
`POST /v1/admin/index/reload` that the promote script calls on each
instance) is a future change.

**The cached value is the whole row**, not just the version string. The
`hnsw_ef_search` default and the `model_version` are read from the same
row; re-reading the row covers all three.

**The cache is per process.** A shared cache (in Redis) would make the
active-index read depend on Redis, which is optional by design; the
active index is not. A failed Redis must not block a promote.

### Evaluation config: never read by the API

`evaluation/thresholds.yaml` is read by `scripts/eval.py` and by CI. The
API does not import it and does not know it exists. The separation is
structural: the file lives under `evaluation/`, which no module under
`src/recsys/api/` imports. A test asserts that the API's import graph
does not reach the evaluation package.

### Precedence: no overlaps

Because each value belongs to exactly one category, there is no
precedence rule to state. A value read from the environment is a boot
config; a value read from `hot.yaml` is a hot config; the two sets are
disjoint. The validation enforces the disjointness in one direction: a
field named in `hot.yaml` that is also present in `Settings` as a
non-derived field is a startup error (the operator has configured the
same value in two places and only one will be read).

The one exception is deliberate. `RATE_LIMIT_PER_MINUTE` exists in
`Settings` as the *default* for any class that `hot.yaml` does not
override. The hot config's per-class values take precedence; the
`Settings` value applies when a class is absent from the file. The
derivation is explicit in code:

```
effective_limit(class) = hot.rate_limit[class] if class in hot else settings.rate_limit_default
```

The rule is "the file is the override, the environment is the default",
stated once and tested.

### Secrets: environment only, never a file, never a log

Secrets — `JWT_SECRET`, `API_KEYS`, `API_KEYS_ADMIN`,
`USER_ID_HASH_SALT`, `DATABASE_URL`'s password — come from environment
variables only. They are never in a committed file, a hot config, a
database row, or a log line.

The prod guard (above) checks that the secrets that must be set are set
and are not the sentinels from `.env.example`. The ADR-0021
forbidden-field list is the rule that keeps them out of logs, metrics,
and traces; the test in `tests/unit/test_log_redaction.py` is the check.

**The `.env` file is a development convenience, not a production
mechanism.** Pydantic Settings reads `.env` when it exists; a production
deployment does not ship a `.env` file, and the value comes from the
environment the orchestrator sets. `.env.example` is committed and
documents every variable; `.env` is git-ignored (M0 `.gitignore`).

**Rotation is a deploy.** A secret change is a `Settings` re-read, which
requires a restart. The two-step rotation procedure (ADR-0013 § API key
storage and rotation) is the only case where a value is added before the
old one is removed; both are in the environment for the duration of the
rotation window.

### Test isolation

Tests do not read the real environment or the file on disk. Two helpers
make this explicit:

- `Settings(_env_file=None, **overrides)` constructs a `Settings`
  instance without reading `.env` and with named overrides. A test that
  needs a specific value passes it; a test that does not gets the
  defaults, not the developer's environment.
- `get_settings.cache_clear()` resets the singleton between tests.

For hot config, a test constructs the loader with a `path` argument
pointing at a `tmp_path` file. The loader is a pure function of the file
content and the polling interval; the thread that drives it in
production is not started in the test.

The two rules are: **a test never reads the machine's state** (an
environment variable, a file at a known path, a clock) unless the test
is about the machine's state, in which case it constructs the state
itself.

## Alternatives considered

| Option | Why not |
|---|---|
| **A single source (environment only)** | A rate limit change during an incident would require a deploy. The hot file exists for the values whose change is operational, not architectural. |
| **A single source (a database)** | Adds a query on the config path and a failure mode (the config is unreachable). The project's operational model is "git is the source of truth for the app's behavior"; a database of config values is a second truth. |
| **A feature-flag service (Unleash, Flagsmith)** | A second external dependency, a second UI, a second SLA. The hot file covers the same use cases for a single-operator project with a committed file. |
| **Hot config for every value** | A `DATABASE_URL` change mid-process would leave the connection pool pointed at the old database and the new value in effect for new connections; the mix is worse than a restart. The criterion for hot config (bounded blast radius) excludes the values whose change should be a deploy. |
| **`inotify` for the hot config watcher** | Linux-only; container bind mounts have unpredictable `inotify` behavior; a 5-second poll is cheap and portable. |
| **No validation of the hot config on change** | A typo in the file would take effect (a rate limit of zero, a negative weight) with no signal. Validation before the swap keeps the previous value and logs the error. |
| **Reload on every request (read the file each time)** | A `stat` + a read on the hot path for values that change once a week. The poller is one `stat` every 5 seconds, off the request path. |
| **Active index read on every request** | A `SELECT` per request against a table that changes only on a promote. The 10-second cache is one query per instance per 10 seconds. |
| **Active index signalled by the promote script (a `POST /v1/admin/index/reload`)** | Requires the script to know every instance's address, which the project does not have (no service discovery). A short poll is the mechanism that does not need the address list. |
| **Precedence rules for overlapping sources** | A value that appears in two places is a configuration bug. Making the sets disjoint is one fewer rule to remember and one fewer way to be surprised. |
| **A `.env` file in production** | The orchestrator sets environment variables; a file on disk is one more thing to mount, one more thing to leak, and one more place for a value to drift from the deployment manifest. `.env` is a development convenience. |
| **Reload secrets without a restart** | A secret reload implies a code path that re-reads the secret and (somewhere) logs the reload event. One read at startup is the moment the secret is in memory; a re-read is a second moment and a second class of bug. |
| **A `Settings` instance per request** | Re-validates every field on every request, and the instance lifetime makes it possible to hold an old value in one code path and a new value in another. The singleton is the process's view of the environment; the environment does not change while the process runs. |
| **Tests reading the real environment** | A test that passes on one developer's machine and fails on another (a stray `INDEX_BACKEND` in the shell) is the worst kind of test. `_env_file=None, **overrides` is one line and removes the class of flakiness. |

## Consequences

**Positive**

- **Every value has one place to look and one rule for changing it.**
  The table at the top of the Decision section is the whole map.
- **A misconfiguration fails at startup.** The process either comes up
  with a valid configuration or does not come up, which is the
  behavior a container orchestrator can act on (restart, alert).
- **The hot config is bounded.** Three classes of values, one file, one
  validation, one log line on a bad change. The values that can be
  changed without a deploy are the ones whose wrong value has a small
  blast radius.
- **Secrets are environment-only and are checked for the sentinel
  values.** The prod guard is the mechanism; the forbidden-field list
  is the follow-through.
- **The active index changes without a restart.** A promote takes
  effect within 10 seconds on every instance; the operator does not
  need to know the instances' addresses.
- **Tests are isolated by construction.** `_env_file=None, **overrides`
  and a `tmp_path` file mean a test's result does not depend on the
  machine it runs on.

**Negative / accepted trade-offs**

- **A hot config change takes up to 5 seconds to be picked up.** The
  rate limit's change is not immediate. For an operator adjusting a
  limit during an incident, 5 seconds is well inside the interval at
  which they would refresh a dashboard to see the effect.
- **The active index takes up to 10 seconds to be picked up.** A
  promote is not instantaneous across instances. The promote script's
  own duration (ADR-0006) is already longer than that; the delay is a
  property of the mechanism, not a surprise.
- **The active index's 10-second TTL is one `SELECT` per instance per
  10 seconds.** At ten instances that is one query per second on a
  small indexed table. A deployment with a hundred instances would want
  a longer TTL or a signal; the number is config and the concern is
  recorded.
- **The prod guard is a list to maintain.** Every secret the project
  adds must be added to the guard (or explicitly exempted). A test
  asserts the guard's list is a subset of `Settings`'s fields, so a
  missing entry is caught rather than silently absent.
- **Hot config is a second file to keep in sync with the code.** A
  field in `hot.yaml` that the code no longer reads is a silent no-op.
  The loader rejects unknown fields (the file's schema is closed), so a
  stale field is a startup error, not a silent one.
- **The hot config does not include secrets by rule.** A deployment that
  wants to rotate a rate-limit-related value that happens to be a secret
  (none exists today) would have to use the environment and a restart.
  The rule is what keeps the file commit-safe.
- **`.env.example` is documentation that can drift.** A new variable
  that is not added to `.env.example` is invisible to a reader of the
  file. A test (`tests/unit/test_env_example.py`) asserts that every
  field in `Settings` appears as a key in `.env.example` and vice
  versa; the drift is caught at PR time.

## References

- `docs/contracts.md` § 3 — the config contract this ADR extends
- `docs/adr/0006-pgvector-schema.md` — the `index_registry` row the
  active index is read from
- `docs/adr/0010-evaluation-thresholds.md` — the evaluation config the
  API does not read
- `docs/adr/0013-authentication-strategy.md` — the two-step secret
  rotation procedure
- `docs/adr/0014-rate-limiting.md` — the rate-limit classes the hot
  config overrides
- `docs/adr/0016-reranker-composition.md` — the re-ranker weights the
  hot config carries
- `docs/adr/0017-experiment-assignment.md` — the static experiment
  declaration this ADR places
- `docs/adr/0021-observability-contract.md` — the forbidden-field list
  the secret rule uses
- `src/recsys/config/settings.py` — the Pydantic Settings model
- `src/recsys/config/hot.py` — the hot config loader and poller
- `config/hot.yaml` — the committed hot config
- `tests/unit/test_settings.py`, `tests/unit/test_hot_config.py`,
  `tests/unit/test_env_example.py`
