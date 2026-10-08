# ADR-0013: Authentication strategy (API key + JWT)

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

The API serves two kinds of callers with different trust models:

- **Server-to-server callers** (internal services, B2B partners, integration
  tests). They need a credential that is stable, machine-readable, and does
  not require a login flow. A shared secret handed to the operator is the
  natural fit.
- **User-facing callers** (mobile and web clients acting on behalf of a
  user). They need a credential that expires, that can be revoked without
  rotating every other caller's secret, and that can carry claims
  (`user_id`, `scope`, `exp`). A signed token is the natural fit.

The M0 scaffold (`docs/contracts.md` § 2, `src/recsys/api/deps.py`) names
both `X-API-Key` and JWT. This ADR fixes what each credential means, how
they are validated, what failure looks like, and how a request is scoped.

## Decision

### Both credentials are accepted, and the caller picks one

A request authenticates by presenting exactly one of:

- `X-API-Key: <key>` — a shared secret. Used by server-to-server callers.
- `Authorization: Bearer <jwt>` — a signed token. Used by user-facing
  callers.

Presenting both is an error (`400`) rather than a silent preference. Two
credentials on one request is a client bug; failing loudly makes it
visible.

### API key storage and rotation

API keys are never stored in plaintext. The config carries hashes
(Argon2id, via `passlib`), and validation is a hash comparison against the
configured set. The `.env.example` uses obvious development values; the
prod guard in `Settings` refuses to boot when `APP_ENV=prod` and
`API_KEYS` is empty.

Rotation is a two-step operator procedure, documented in
`docs/ops.md` § Key rotation:

1. Add the new key hash to `API_KEYS` and deploy. Both keys now work.
2. Remove the old key hash and deploy. Only the new key works.

The two-step procedure exists so that rotation does not require an
instant cutover. A one-step rotation would invalidate every in-flight
caller at the moment of deploy, which for a B2B caller with a long-lived
connection is a request-visible outage.

### JWT

- **Algorithm:** HS256 by default, configurable via `JWT_ALGORITHM`. RS256
  is supported by configuration but is not the default for a single-node
  deploy with no external identity provider.
- **Secret:** from `JWT_SECRET`, never from a default. The prod guard
  refuses to boot when the secret is the sentinel value.
- **Claims required:**
  - `sub` — the subject (a user id, an opaque string).
  - `exp` — expiration time. A token without `exp` is rejected, not
    accepted-with-warning.
  - `iat` — issued at. Used for the maximum-age check below.
  - `scope` — a space-separated list of scopes; empty or absent means
    "no scopes".
- **Maximum age:** a token whose `iat` is older than
  `JWT_MAX_AGE_SECONDS` (config; default 24 hours) is rejected even if
  `exp` is in the future. This bounds the damage from a leaked token
  whose owner set a very long `exp`.
- **`kid` header:** supported. On rotation, the operator adds a new
  secret under a new `kid`, deploys, then removes the old one. The
  validator picks the secret by `kid`. A token whose `kid` is unknown is
  rejected.

### Scopes

Two scopes are defined for M3:

| Scope | Grants |
|---|---|
| *(none)* | Read access: `/v1/recommend`, `/v1/items/{id}/similar`, `/v1/events` |
| `admin` | The above, plus privileged operations (index promote, rollback; M6) |

The scope set is not extensible by callers. A future ADR adds a scope;
the code's `SCOPE_*` constants are the single source of truth.

An API key authenticates as **no scope** unless it appears in a separate
`API_KEYS_ADMIN` list. A key in `API_KEYS_ADMIN` grants the `admin` scope
in addition to read. The reason for the separate list rather than a
prefix or an embedded scope: a leaked read key must not become an admin
key through a config edit that forgot to remove it from the wrong list.

### Fail-closed on infrastructure failure

Authentication never falls back to "allow" when the validator itself
fails. If the Redis-backed revocation list is unreachable (see the
revocation section below), the request is rejected with `503` and a
structured log line; it is not authenticated by default.

The alternative — fail-open with a warning — is rejected. A single
misconfigured environment variable or a transient Redis outage would
silently turn every request into an unauthenticated request, which is the
worst possible failure mode for an auth system. The correct behavior when
the auth system cannot answer the question "is this caller allowed?" is to
refuse, log loudly, and let the operator restore Redis.

### Revocation

M3 does not implement per-token revocation (a JWT deny-list, a per-key
deny-list, or a session store). The reason: with HS256 and short-lived
tokens, the operationally correct way to revoke a JWT is to rotate the
signing secret (which invalidates every token) or wait for `exp`. A
deny-list adds Redis on the auth hot path for a use case the project does
not yet have.

If per-token revocation becomes necessary, a new ADR adds it, and the
fail-closed rule above applies: a deny-list that cannot be read means the
request is rejected.

### Rate limit keyed by credential

The rate limiter (ADR-0014) keys on a **hash of the presented credential**,
not on the client IP. Two properties drive this:

- **IP is not the caller.** A single API key used from many IPs is one
  caller; many callers behind one NAT are many. Keying on IP punishes the
  wrong party and is trivially bypassed by rotating source addresses.
- **The credential is the identity we authenticate.** If we trust it
  enough to authorize the request, we trust it enough to meter it.

The rate limiter never stores the raw credential; it stores a BLAKE2b
hash prefixed with a fixed namespace.

### Failure responses

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Both credentials presented, or a malformed `Authorization` header |
| 401 | `unauthenticated` | Missing credential, unknown key, invalid signature, expired token, `iat` too old, unknown `kid` |
| 403 | `forbidden` | Credential is valid but the scope is insufficient for the route |
| 503 | `unavailable` | The auth validator itself could not run (see fail-closed above) |

The distinction between 401 and 403 is enforced: a caller with a valid
credential but no scope receives 403, not 401. A caller who presents no
credential receives 401, not 403. Log lines carry `reason`
(`missing`, `invalid`, `expired`, `iat_too_old`, `unknown_kid`, `no_scope`)
but never the credential itself.

### No credential in logs, metrics, or errors

Raw API keys, raw JWTs, and raw `Authorization` headers never appear in:

- structlog output (any level, any field).
- Prometheus metric labels (the metric label is the credential hash,
  truncated, and only for the rate limiter's own counters).
- Error messages returned to the caller.
- Trace attributes.

The rule is mechanical: the credential is read once, from a header, into a
local variable; the validator produces a `Principal` (subject + scopes +
credential-hash); the header value is discarded. Everything downstream
sees the `Principal`, not the credential.

## Alternatives considered

| Option | Why not |
|---|---|
| **API key only** | No expiration, no per-caller revocation, no claims. A leaked key is valid until an operator rotates it. Fine for internal callers, wrong for user-facing ones. |
| **JWT only** | Server-to-server callers would have to acquire a token before every request, which is a login flow for a machine. Adds a component (the token issuer) with no benefit for the B2B case. |
| **OAuth2 / OIDC with an external provider** | Correct for a multi-tenant product, overkill for a single-node portfolio project. The external provider becomes a hard dependency with its own availability target. |
| **Store API keys in plaintext** | A database dump or a config leak becomes a credential leak. Argon2id is not expensive to verify (a few milliseconds) and removes the "our secrets are in the clear" failure mode. |
| **Fail-open when the validator fails** | Turns a Redis outage into an authentication bypass. Rejected: the correct answer to "cannot determine authorization" is to refuse. |
| **Per-token deny-list in M3** | Adds Redis to the auth hot path for a use case the project does not yet have. Rotation + short `exp` is the operationally correct revocation for a single-operator deploy. A future ADR adds the deny-list when a caller actually needs instant per-token revocation. |
| **Rate limit per IP** | IP is not the caller. Punishes NATs, misses distributed abuse from one credential across many IPs, and is trivially bypassed. |
| **Combine admin and read keys in one list with a prefix (`admin:...`)** | A misconfigured prefix rule can silently promote a read key. A separate list is unambiguous; the operator cannot add a key to the wrong list by getting a prefix wrong. |
| **Return 401 for missing scope** | Conflates "who are you" (401) with "you may not do this" (403). Clients that retry with a fresh token on 401 would retry forever against a 403 situation. |

## Consequences

**Positive**

- **Two credential models for two caller types.** Server-to-server uses a
  stable shared secret; user-facing uses a signed token with claims. The
  caller picks the one that fits; the API does not require either party to
  adopt the other's model.
- **Fail-closed is stated as a rule, not as a default.** Every future
  addition to the auth path (deny-list, key registry, OAuth introspection)
  inherits the rule. The failure mode "Redis is down and everyone is now
  unauthenticated" cannot happen.
- **Rotation is a documented two-step procedure.** In-flight B2B callers
  are not dropped mid-request by a rotation deploy.
- **401 and 403 mean different things.** Clients can react correctly: 401
  means refresh the credential; 403 means ask for a different scope.
- **The credential never appears in observability data.** A leaked
  dashboard or log dump does not leak credentials.

**Negative / accepted trade-offs**

- **Argon2id verification costs a few milliseconds per request.** On a
  request that already spends ~10 ms in retrieval, the relative cost is
  small, but it is not zero. Mitigation if it becomes a problem: an LRU
  cache of recently verified keys (with a short TTL) at the auth layer.
  Not implemented until measured.
- **Two API key lists (`API_KEYS` and `API_KEYS_ADMIN`) is one more thing
  to configure.** The trade-off is deliberate: a single list would need a
  scope marker per key, and the failure mode of mis-marking a key is
  privilege escalation. Two lists make the mistake mechanical (the key is
  in the wrong list) rather than subtle (the key has the wrong prefix).
- **No revocation until a future ADR.** A leaked JWT is valid until `exp`
  or until the signing secret is rotated. For a single-operator project
  this is acceptable; for a team, a deny-list would be worth the
  complexity. The decision is recorded so it can be revisited.
- **Rotation requires two deploys.** The alternative is a hot-reload of
  the key set, which is one more moving part (watched file, SIGHUP, or a
  config service) than M3 needs. M6 can revisit if the deploy cadence
  becomes painful.
- **`iat` maximum age is an additional rejection rule that not every JWT
  issuer sets.** A token without `iat` is rejected. This is a deliberate
  narrowing: a token that cannot be bounded in age is a token that cannot
  be safely accepted.

## References

- `docs/contracts.md` § 2 — the API contract this ADR implements
- `docs/adr/0014-rate-limiting.md` — rate limiting keys on the credential
  hash this ADR defines
- `docs/ops.md` § Key rotation — the two-step rotation procedure
- `src/recsys/api/deps.py` — the FastAPI dependency that produces the
  `Principal`
- `src/recsys/config/settings.py` — `API_KEYS`, `API_KEYS_ADMIN`,
  `JWT_SECRET`, `JWT_ALGORITHM`, `JWT_MAX_AGE_SECONDS`
- `docs/runbook.md` — Runbook 5, auth failures (added in M3.7)
