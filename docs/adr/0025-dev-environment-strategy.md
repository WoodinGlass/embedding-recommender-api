# ADR-0025: Development environment strategy

- **Status:** Accepted
- **Date:** 2026-10-09
- **Deciders:** project maintainer

## Context

The project is developed in two environments that are not the same:

- **Google Colab** — a notebook session, no root, no Docker daemon,
  no local PostgreSQL. It is where code is written and where the
  fast feedback loop lives.
- **GitHub Actions** — the only environment that runs the container,
  the pgvector-backed evaluation job, and the encoder parity check.

Until M3.4 the gap between the two was papered over with a stopgap:
a GitHub personal access token stored as a plaintext file in Google
Drive. Three failure modes drove a rework:

1. **Secrets in Drive are one accidental share away from exposure.**
   Drive links are searchable, and a shared folder can carry a token
   to someone who never saw the repository.
2. **The failure loop is slow when DB behavior is only exercised in
   CI.** Two bugs in M3.4 — a psycopg transaction left in the
   aborted state and a pgvector text column parsed as a float list —
   were invisible in Colab and surfaced as CI reds minutes apart.
   Each iteration cost a full CI round trip.
3. **Tool version drift between local, the pre-commit hook, and CI
   is not visible.** Ruff was pinned at `0.6.9` in one place and
   `>=0.6,<1.0` in another; the same commit could pass in one
   environment and fail in another for reasons that have nothing to
   do with the code.

The strategy below is deliberately not "run CI locally". Colab cannot
run a Docker daemon or a local PostgreSQL, so pretending otherwise
would leave the DB path untested at the moment it is written.

## Decision

### Secrets: Colab Secrets for live values, an encrypted file for the rest

- **GitHub PAT** lives in **Colab Secrets** (sidebar key icon). It
  is a session credential, not a project artifact; encrypting it
  would only move the problem. The setup cell reads it with
  `google.colab.userdata.get('GH_PAT')` and exports it for git's
  askpass helper. Nothing writes it to disk.
- **Everything else** (`DATABASE_URL`, and in M3.6
  `RECSYS_TEST_REDIS_URL`) lives in a **Fernet-encrypted file**
  `env.enc` in Drive. The Fernet key lives in Colab Secrets as
  `SECRETS_KEY`. Drive holds ciphertext; a leaked Drive file without
  the key is inert. One secret to manage in Colab, not N.
- **Fernet key padding.** `Fernet.generate_key()` returns 44
  characters including a trailing `=`; Colab Secrets strips that
  character on save. The load cell pads the value back to a multiple
  of four before constructing the `Fernet`. This is documented in
  the load cell and is the reason the loader checks length rather
  than assuming.

### Development loop: pre-commit, one ruff version, `--ff`

- **`.pre-commit-config.yaml` runs ruff (lint + format), mypy, and
  housekeeping checks on every commit.** The hook environment is
  ephemeral (per Colab session) and the hook file itself is in Drive.
- **One ruff version, in three places.** `pyproject.toml`,
  `.pre-commit-config.yaml`, and the version CI installs must be
  identical. A mechanical check
  (`scripts/check_ruff_version.py`, wired into `make check` and CI)
  fails the build when the installed ruff does not match the pin.
  The check exists because version drift is invisible without it.
- **`make test-unit` runs pytest with `--ff`.** On a fresh checkout
  the flag is a no-op; on a session where a test just failed it runs
  that test first. Nothing about which tests run changes.

### DB behavior: CI is the verification environment, not a fallback

This is the part that is easiest to misread, so it is stated
explicitly:

- **Integration tests against PostgreSQL and Redis run in CI.**
  They do not run in Colab. `tests/integration/**` skips cleanly
  when `RECSYS_TEST_DATABASE_URL` is not set, which is the Colab
  state.
- **A local Postgres via a managed service (Neon) is optional, not
  required.** If a developer sets it up, it is for a smoke test, not
  for the real gate. The version of pgvector on the managed service
  will not match the `pgvector/pgvector:pg16` service container in
  CI, and `SELECT ... FOR UPDATE` behavior depends on the version
  and the pool configuration. A green run on Neon is not a green
  run on CI, and treating the two as equivalent would make the gate
  weaker, not stronger.
- **The reverse is also true:** a red run on CI when Neon is green
  is not a Neon bug. It is the CI environment saying something about
  the code. Diagnose it there.

### The fast feedback loop stays fast

The combination above is chosen so that:

- A change to pure logic (most of the code) is verified in Colab in
  seconds by `make check`.
- A change that touches the database is verified in CI in one round
  trip. This is the cost, and it is accepted rather than hidden.
- The number of DB-touching changes is bounded by keeping the DB
  surface small (repositories with a narrow interface, most logic
  above them pure).

## Alternatives considered

| Option | Why not |
|---|---|
| **Plaintext secrets in Drive** | One accidental share away from exposure; Drive's link sharing is not opt-in per file. |
| **Secrets in `.env` committed to the repo** | Universal anti-pattern; `.env` is gitignored, and the setup exists precisely because a gitignored file cannot be shared between two machines. |
| **All secrets in Colab Secrets, none in Drive** | Colab Secrets is per-session and per-notebook. A developer who opens a new notebook to reproduce an issue starts from zero; an encrypted file in Drive survives. The split (live credential → Secrets, rest → encrypted file) matches the persistence each secret needs. |
| **Fernet key in Drive next to the ciphertext** | The ciphertext and the key travel together; that is a plaintext secret with extra steps. |
| **`pre-commit` framework vs a hand-written shell hook** | A hand-written hook would use the same ruff the shell uses, avoiding a separate environment. It was rejected because the framework runs other useful checks (trailing whitespace, YAML/TOML parse, private-key detection) that a hand-written hook would have to reimplement, and because the hook environment's isolation is a feature: a broken developer environment does not silently disable the checks. The version pin handles the drift the framework would otherwise introduce. |
| **Managed Postgres (Neon) as a required dev dependency** | Adds a network hop, a quota, a version skew from the CI service container, and ongoing cost, to replace an integration suite that already runs in CI. A smoke test is not the same as a gate, and treating it as one would make the gate weaker. |
| **Run CI jobs on a self-hosted runner so it is "the same machine"** | Out of scope for a single-maintainer project; the maintenance cost of a runner exceeds the CI minute cost it saves. |
| **Skip ruff pinning and let each environment pick** | The M3.4 commits demonstrate the failure: PT011 and PT018 were flagged in one environment after passing in another. Pinning and checking mechanically removes the class. |
| **Encrypt each secret separately with its own key** | More secrets in Colab Secrets, more code paths, and no additional protection: the Fernet key already protects every value in the file. A single file with one key is simpler and equally safe. |

## Consequences

**Positive**

- **No plaintext secret in Drive.** A leaked `env.enc` is inert
  without a key that lives in a service the same Drive does not
  control.
- **Ruff version drift is caught by the build, not by a red CI on a
  commit that passed locally.** The check is the missing piece that
  made the pin meaningful.
- **The DB gap is stated, not implied.** A reader knows that a
  green Colab session is not a green integration suite, and that
  CI is not a slower local environment but the environment where
  that class of bug is designed to surface.
- **The feedback loop for pure logic is unchanged.** Most changes
  do not touch the DB; those changes still get a sub-second lint
  and a multi-second test.

**Negative / accepted trade-offs**

- **The first commit of a Colab session installs the hook
  environments again.** Roughly 30 seconds. A developer who does
  not plan to commit does not pay it (`make install-hooks` is not
  run automatically).
- **The encrypted file has one more step than a plaintext file.**
  A new developer must obtain `SECRETS_KEY` from the maintainer
  (or generate a new one and start a new file). For a
  single-maintainer project this is the maintainer; for a team it
  would need a sharing mechanism the current setup does not
  provide.
- **A Fernet key lost is a file lost.** There is no recovery path:
  the ciphertext cannot be decrypted without the key, and Drive
  does not version `.secrets/` by default. The setup cell says so.
- **The ruff check does not cover `.pre-commit-config.yaml`'s
  `rev:` line.** The check reads the pin from `pyproject.toml` and
  the installed version; the pre-commit `rev:` is checked only by
  review, because the framework does not expose the resolved
  version to the project environment. A future improvement is to
  parse `.pre-commit-config.yaml` in the same script and assert
  all three agree; it is not in this ADR because the file format
  is not part of the project's contract and could change.
- **DB-heavy milestones (M3.5, M3.6) pay a CI round trip for every
  DB bug.** This is deliberate: the alternative was a managed
  service whose version skew from CI would hide exactly the class
  of bug that motivated the ADR.

## References

- `docs/adr/0012-backend-abstraction.md` — the backend protocol
  the integration tests exercise
- `docs/adr/0015-cache-and-circuit-breaker.md` — the cache and
  breaker whose integration tests run in CI, not in Colab
- `docs/adr/0016-reranker-composition.md` — the re-ranker whose two
  M3.4 bugs drove this ADR
- `.pre-commit-config.yaml` — the hooks and their pinned ruff
- `scripts/check_ruff_version.py` — the mechanical check
- `docs/contracts.md` § 6 — change management
