# ADR-0018: Event ingestion

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

`POST /v1/events` accepts `impression`, `click`, and `conversion` events.
Two consumers depend on this stream:

- **M5's experiment analysis** reads exposures and outcomes to compute
  SRM, significance, and guardrails. A missing or duplicated event is a
  wrong denominator or a phantom trial.
- **M7's churn model** reads the same stream to build recency, frequency,
  and engagement-trend features. A gap in the stream is a gap in the
  features.

`docs/contracts.md` § 1.1 fixes the `EventEnvelope` schema and the
idempotency rule (`event_id` unique at ingest). What it does not fix:

1. **Single event or batch.** One HTTP request per event is the shape the
   scaffold implies; it is also the shape that makes a mobile client
   sending a session's worth of interactions at once issue one request
   per interaction.
2. **How the user id is stored.** `contracts.md` § 1.1 says the server
   hashes the `user_id` before storage; the algorithm and the salt
   policy are not stated.
3. **What counts as an acceptable timestamp.** Late events are normal
   (offline queue); arbitrarily late events are a sign of a
   misconfigured client or a replay.
4. **Retention.** An append-only stream grows without bound.
5. **Synchronous vs asynchronous write.** M3 has no queue; the scaffold
   implies a direct insert.

## Decision

### The endpoint accepts a batch of 1 to 100 events

```
POST /v1/events
Content-Type: application/json

{
  "events": [ { ...envelope... }, ... ]   # 1..100
}
```

Response `202 Accepted`:

```json
{
  "accepted": 95,
  "duplicates": 5,
  "rejected": 0
}
```

**Why a batch.** A mobile client that collects a session's interactions
and sends them at the end of the session issues one request for the
session, not one per interaction. At 100 events per session the
difference is two orders of magnitude in request count. The server-side
cost of a batch is a single `executemany` against the same table a single
insert would hit.

**Why max 100.** A batch of 1 000 has a worst-case request body of
roughly 500 KB (an envelope is ~500 bytes) and a worst-case transaction
duration that is hard to bound on a slow database. At 100 the body is
~50 KB and the transaction is short enough to reason about. A caller
with more events sends more requests; the rate limit for the `events`
class (ADR-0014) is set with this in mind.

**Why 1 is allowed.** A single event is the batch of one. Requiring a
list even for one event keeps the schema uniform; the caller does not
have to know which shape the server is in a mood for.

**Single-event callers from the M2 scaffold.** The M2 `EventEnvelope`
was a top-level object. The M3 endpoint takes a list. The migration is
mechanical for a caller: wrap the object in `[ ... ]`. The
`docs/contracts.md` § 1.1 example is updated in the same PR; the API is
not yet in production.

### Idempotency: client-supplied `event_id`

Every event carries an `event_id` the client generates (a UUIDv4, or any
unique opaque string). The database has a unique constraint on
`event_id`; the insert is `INSERT ... ON CONFLICT (event_id) DO NOTHING`.
A duplicate is counted in the `duplicates` field of the response and
ignored otherwise.

**If `event_id` is absent** (a client that has not adopted the field, or
a manual call during development), the server generates one and returns
it in the response's `generated_event_ids` array (parallel to the input
array). The generated id is `sha256(f"{request_id}:{index}")`, which is
stable for a given request and index; a retry of the same request
produces the same ids and the insert is idempotent. A client that
retries with a different `request_id` gets a different set, which is
correct: the client asked for new events to be recorded, twice.

**Why not dedup by content.** Two identical clicks at the same second on
the same item are a real possibility (a user double-taps) and should be
recorded twice. Content-based deduplication would collapse them; the
`event_id` a client generates reflects the client's intent, which is the
right level of granularity.

### Timestamp skew window

`event_ts` is accepted when it is within `[now - 7 days, now + 5 minutes]`.
Outside that range the event is rejected with `422` and an error detail
naming the field and the offending timestamp.

- **7 days back** covers an offline queue on a mobile client that has
  been disconnected for a week. Beyond a week, an event is more likely
  to be a replay or a clock problem than a real interaction.
- **5 minutes forward** covers normal clock skew between a client and
  the server. A client whose clock is more than 5 minutes fast is a
  misconfiguration the caller should fix.

The window is config (`EVENT_TS_MAX_AGE_SECONDS`,
`EVENT_TS_MAX_FUTURE_SECONDS`), not a literal. The values above are the
defaults.

**Rejecting a late event rather than clamping it.** A "clamp to now"
policy would silently misattribute a week-old event to the present,
which corrupts both the experiment analysis (the exposure and the click
would be in the wrong order) and the churn features (the recency term
would be wrong). Rejecting is the honest failure: the client learns its
event was not recorded and can decide what to do.

### User id hashing: HMAC-SHA256 with a versioned salt

The client sends `user_id`. The server stores `user_id_hash` computed as:

```
user_id_hash = HMAC_SHA256(key=user_id_salt, message=user_id)
```

and it also stores `user_id_salt_version`, an integer that names which
salt was in use.

**Why HMAC, not SHA256.** A plain SHA256 of a `user_id` is reversible
against a dictionary when the id space is small or predictable (an
email address, a phone number, a sequential id). HMAC with a secret key
removes that attack: an attacker with the hashes but not the salt cannot
enumerate the input space.

**Why a salt version.** Rotating the salt (a key compromise, an annual
rotation, a change in the id format) produces different hashes for the
same `user_id`. Without a version marker, the analysis would see one
user as two. With it, the analysis can filter on a single salt version
or join across them knowing the rotation date.

**Where the salt comes from.** `USER_ID_HASH_SALT` and
`USER_ID_HASH_SALT_VERSION` in config, sourced from environment
variables. The prod guard in `Settings` refuses to boot when
`APP_ENV=prod` and the salt is a sentinel or the version is `0`.

**Where the hash is computed.** In the event-ingestion handler, after
Pydantic validation and before the database insert. The raw `user_id`
is a local variable in that handler and is not passed to any code that
logs, traces, or stores it. The `EventEnvelope` Pydantic model carries
the raw `user_id` because validation needs it; the model's
`model_dump` is not called on the way to storage.

**Where the raw `user_id` may not appear.** Anywhere:

- structlog output, at any level.
- Prometheus labels (the label is the hash, truncated, if used at all).
- Trace attributes.
- Error messages returned to the caller.
- The database.

### PII redaction is structural, not incidental

The rule is enforced by the shape of the code, not by a promise to be
careful:

- The event-ingestion module is the only place in `src/recsys/` that
  reads `envelope.user_id` from a validated model.
- The raw value is passed only to the hashing function, whose return
  value is what the module uses thereafter.
- The `EventEnvelope` Pydantic model's `__repr__` (Pydantic's default)
  prints every field; the module never calls `repr` or `str` on an
  instance. A test asserts that a log line written during ingestion does
  not contain a known `user_id` sentinel.
- A `tests/unit/test_pii_redaction.py` runs the ingestion path with a
  distinctive `user_id` value and asserts the value does not appear in
  captured stdout, stderr, or the recorded SQL parameters.

### Synchronous write in M3

The handler validates, hashes, and inserts in one database transaction,
then returns `202` with the counts. There is no queue, no worker, no
outbox.

**Why synchronous.** M3's traffic target is modest, the write is a
single `executemany` into an indexed table, and a synchronous write
gives the caller an accurate count in the response. A queue would make
the response report "accepted" for events that a worker might later
fail to write, which is a weaker guarantee than the caller can use.

**Why this is a documented deferral, not an oversight.** At a traffic
level where the synchronous insert dominates the request, the correct
answer is a queue (Redis Streams, Kafka, or an outbox pattern with a
relay). M4's load test measures the write's contribution to latency; if
it becomes material, an ADR adds the queue and the response becomes a
provisional acknowledgment with a separate "committed" signal. Not now.

### Retention

Events are retained for `EVENT_RETENTION_DAYS` (config, default 365).
A scheduled job (M6) deletes rows older than the window. M3 does not
run the job; it documents the policy and the query that implements it,
and `docs/ops.md` § Event retention records the procedure.

The retention window is longer than any experiment the project expects
to run (an experiment at this scale reaches its sample size in days, not
months) and shorter than the disk budget on the single-node target. A
future ADR changes it if either assumption moves.

## Alternatives considered

| Option | Why not |
|---|---|
| **Single event per request** | A mobile session with 100 interactions issues 100 requests. The batch is one request. The server cost of a batch is one `executemany`; the client cost is one round trip. |
| **Unbounded batch size** | A worst-case body of hundreds of kilobytes and a transaction whose duration depends on the client's payload. The 100-event cap bounds both; a caller with more sends another request. |
| **Plain SHA256 of `user_id`** | Reversible against a dictionary when the id space is small or predictable. HMAC with a secret salt removes that attack at no operational cost. |
| **A per-request salt (never stored) or no hash at all** | The analysis needs to group events by user; a non-reproducible hash cannot group, and storing the raw id stores PII. A versioned secret salt is the middle: reproducible, non-reversible, and rotatable. |
| **Clamp a late timestamp to `now`** | Silently misattributes an old event to the present, corrupting the exposure-before-outcome ordering the experiment analysis depends on. Rejecting is honest. |
| **No skew window at all** | A client with a badly wrong clock writes events dated in 1970 or 2099, which sorts incorrectly in every downstream query. The window is a cheap guard. |
| **Content-based deduplication** | Collapses two real double-taps into one. The `event_id` reflects the client's intent, which is the right granularity. |
| **Idempotency key from the server only (ignore a client-supplied `event_id`)** | A client that retries an accepted-but-lost request has no way to say "this is the same event". A server-generated id is a different id on the retry. |
| **Server-generated ids returned as the response body's primary payload (an array of new ids for every event)** | Turns a fire-and-forget ingest into a caller that must correlate a returned id with its original event. The `event_id` field on the input is the correlation; a generated id is a fallback for callers that omit it. |
| **Asynchronous write via a queue in M3** | Adds a component (the queue) and a failure mode (the queue is full) for a write the target hardware can absorb synchronously. The correct moment to add a queue is when a measurement shows the synchronous write is the bottleneck. |
| **A separate `POST /v1/events/batch` for batches** | Two endpoints, two schemas, two code paths, one behavior. The single endpoint with a 1..100 list covers both cases with one shape. |
| **No retention policy** | An append-only stream grows without bound; on the single-node target the disk fills and the API stops. The retention window and the delete query are documented even though the scheduled job lands in M6. |
| **Soft delete (a `deleted_at` column) for events** | The analysis does not need deleted events, and keeping them doubles the table's steady-state size for a use case nobody has. A hard delete after the retention window is what the policy means. |

## Consequences

**Positive**

- **One request covers a session.** A mobile client sends its batched
  session once; the server inserts the batch in one transaction.
- **Idempotency is free for a well-behaved client.** A client-supplied
  `event_id` makes retry safe without a second lookup or a distributed
  lock.
- **PII never reaches storage, logs, or traces.** The HMAC hash is the
  only user identifier the system keeps, and the salt version makes a
  rotation a filter rather than a data loss.
- **The skew window and the retention window are stated numbers.** A
  reviewer can reason about both without reading a constant in the
  handler.
- **The synchronous write gives the caller an accurate count.** The
  response reflects what the database holds, not what a queue accepted.

**Negative / accepted trade-offs**

- **A batch of up to 100 events is one rate-limit token.** A caller that
  batches aggressively gets more events per token than a caller that
  sends one at a time. The `events` class limit (ADR-0014) is set with
  the batch in mind; if the effective event throughput becomes a
  problem, a future ADR changes the cost of a batch to scale with its
  size.
- **A late event is lost, not recorded.** A client whose queue is more
  than 7 days stale is rejected. This is correct for the project's
  consumers (both analyses care about recency) but is a policy a
  different product might set differently. The window is config.
- **The salt version is an operator-managed integer.** A rotation that
  forgets to bump the version produces two hashes under one label and
  the analysis sees a single user as two. The startup log prints the
  version so a mismatch is visible.
- **The synchronous write adds the insert's latency to the response.**
  A single `executemany` on the target PostgreSQL is a few milliseconds;
  M4's load test measures the actual contribution. If it becomes
  material, the queue lands per the deferral above.
- **No retention job in M3.** The policy is documented and the query
  exists; the job that runs it lands in M6. Until then, the events table
  grows, which is acceptable for a portfolio project's traffic but is a
  real deferral.
- **A `user_id` that is already hashed by the client is hashed again.**
  The server has no way to know the client's value is already a hash.
  The convention is that the client sends the raw id; a client that
  sends a hash produces a stable (if double-hashed) identifier, which
  still groups correctly. Documented in `docs/contracts.md` § 1.1.

## References

- `docs/contracts.md` § 1.1 — the `EventEnvelope` schema this ADR
  surrounds
- `docs/adr/0014-rate-limiting.md` — the `events` rate-limit class
- `docs/adr/0017-experiment-assignment.md` — the exposure events written
  to the same store
- `docs/ops.md` — the retention procedure and the salt-rotation
  procedure
- `src/recsys/api/routers/events.py` — the endpoint
- `src/recsys/api/schemas/events.py` — `EventEnvelope` and `EventBatch`
- `src/recsys/events/hashing.py` — the HMAC hasher
- `tests/unit/test_pii_redaction.py` and
  `tests/integration/test_event_ingestion.py`
