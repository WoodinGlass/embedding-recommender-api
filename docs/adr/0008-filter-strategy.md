# ADR-0008: Filtered ANN search — iterative scan, fallback, and version detection

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

The retrieval path must serve `POST /v1/recommend` with optional metadata
filters (`category`, `brand`, `language` — see `docs/contracts.md` § 2.1).
The naive approach — HNSW search over the whole table, then discard rows
that fail the filter — has a well-known failure mode: when the filter is
selective, the ANN search returns `k * some_factor` candidates but only a
handful satisfy the filter, so the caller receives fewer than `k` results
even though the catalog contains many matching items.

pgvector 0.8 introduced `hnsw.iterative_scan`, which continues the HNSW
graph walk until enough rows survive the filter. It has three modes:

- `off` — the pre-0.8 behavior. Post-filter only.
- `relaxed_order` — continues scanning; results may be returned out of
  distance order and need re-sorting.
- `strict_order` — continues scanning; results are guaranteed in distance
  order, at the cost of more graph work.

The database in this project is PostgreSQL 16 with the pgvector extension.
The version of that extension is not controlled by `pyproject.toml` — it is
whatever the operator's `CREATE EXTENSION` produced. In CI, the service
container `pgvector/pgvector:pg16` currently ships 0.8.x; a developer
running an older local image may have 0.7.x.

Two things must be true:

1. Filtered queries should return `k` results when the catalog contains
   `k` matching items. Otherwise the API contract (`len(items) <= k`, but
   "may be smaller when filters are very selective" in `docs/contracts.md`
   § 2.1) is technically satisfied but practically useless.
2. A behavior difference between environments must be **visible**, not
   silent. If a query behaves differently on pgvector 0.7 than on 0.8, the
   evaluation report must record which one produced it.

## Decision

### Detect the extension version once, at startup

On the first connection that builds or uses the retrieval layer, query
`SELECT extversion FROM pg_extension WHERE extname = 'vector'`. Parse the
major.minor prefix. Record the raw string for storage in
`index_registry.pgvector_version` (per ADR-0007) and the parsed
`(major, minor)` for runtime decisions.

Log the detected version exactly once per process at `INFO`. If the version
is below 0.8, additionally emit one `WARNING`:

```json
{
  "event": "retrieval.pgvector.old_version",
  "detected": "0.7.4",
  "required_for_iterative_scan": "0.8",
  "fallback": "post_filter_k_multiplier"
}
```

Do **not** re-query the extension version per request. The version does not
change while a connection is alive, and a per-request lookup wastes a
round-trip. If the operator upgrades pgvector, they restart the service
(and, per ADR-0007, rebuild the index if the upgrade is expected to affect
recall).

### Query strategy when the extension supports iterative scan (>= 0.8)

When the request carries filters, the query becomes:

```sql
SET LOCAL hnsw.iterative_scan = 'strict_order';
SET LOCAL hnsw.max_scan_tuples = 20000;

SELECT e.item_id, 1 - (e.vector <=> %(query)s) AS score
FROM embedding e
JOIN item i ON i.item_id = e.item_id
WHERE e.index_version = %(index_version)s
  AND i.category = %(category)s   -- only if the filter is present
ORDER BY e.vector <=> %(query)s
LIMIT %(k)s;
```

- `strict_order` is chosen because the API contract (`docs/contracts.md`
  § 2.1) guarantees that `items` is sorted by score descending, ties broken
  by `item_id`. `relaxed_order` would require an explicit re-sort and would
  make the tie-breaking rule harder to enforce; the latency cost of
  `strict_order` is measured in M2.4 and recorded in the design doc.
- `max_scan_tuples` is bounded so that a pathological query (a filter that
  matches almost nothing) cannot scan the entire table. The current value
  (20000) is a starting point; M2.4 will tune it against the sample catalog
  and the golden set.
- `SET LOCAL` is scoped to the transaction, so it does not leak into
  subsequent queries on the same pooled connection.

When the request carries no filters, `iterative_scan` is not set. Plain HNSW
search is faster and the filter problem does not apply.

### Query strategy when the extension does not support iterative scan (< 0.8)

The same filtered query runs, but with an over-fetch multiplier:

```
effective_k = k * 4
```

Results are post-filtered in Python: the query returns up to `effective_k`
rows, the filter is applied (again, defensively — the SQL `WHERE` clause
still runs), and the first `k` are returned. Each fallback invocation logs
at `WARNING`:

```json
{
  "event": "retrieval.pgvector.fallback",
  "reason": "pgvector_lt_0_8",
  "detected": "0.7.4",
  "requested_k": 10,
  "effective_k": 40
}
```

The multiplier is a constant in this codebase, not a config knob. If it
proves wrong — for example, if a filter matching 1% of the catalog needs a
larger factor — the code is changed deliberately, with an ADR. A config knob
would move the decision out of review.

The `effective_k` and the fallback reason are recorded in the evaluation
report (see `docs/retrieval-and-evaluation.md` § Evaluation). Reports
generated under different pgvector versions are marked as such; comparing
them is comparing apples to oranges and the metadata makes that explicit.

### Filter allowlist from `contracts.md`

The set of filter fields is defined in `docs/contracts.md` § 2.1:
`category`, `brand`, `language`. The retrieval layer reads this set from a
single source — a `FILTER_FIELDS` constant in
`src/recsys/retrieval/filters.py`, which is referenced by both the API
schemas and the retrieval layer. It is not duplicated.

Filter values are always passed as SQL parameters. No string interpolation,
no f-string building of `WHERE` clauses. A filter value that fails the
allowlist is rejected at the API layer with `422` before it reaches
retrieval.

### Filter selectivity and `hnsw.max_scan_tuples`

A filter that matches a large fraction of the catalog (say, `category` with
two values) needs almost no extra scanning. A filter that matches a very
small fraction (say, one brand carried by 3 items) can force a long scan
before `k` results are found. `max_scan_tuples = 20000` bounds this: beyond
20000 tuples scanned, the query returns what it has. If it returns fewer
than `k`, the caller receives fewer than `k` — which the contract permits.
M2.4 will measure the point at which this bound bites on the sample catalog
and adjust if needed.

## Alternatives considered

| Option | Why not |
|---|---|
| **Post-filter only (no iterative scan)** | Silent under-return on selective filters. The API would return `k=10` requested, `k=2` returned, with no explanation. Fine for a toy project, not fine for a system whose whole point is to be measured. |
| **`SET hnsw.iterative_scan` globally, in the pool** | Leaks the setting into queries that do not need it and makes it hard to change per-query. `SET LOCAL` in the transaction is the correct scope. |
| **`relaxed_order` for lower latency** | Would require the retrieval layer to sort results in Python before returning them, and would complicate tie-breaking. The latency difference between `relaxed_order` and `strict_order` on our scale is measurable but small; the correctness benefit of `strict_order` is larger. Documented as a tunable if M2.4 finds the difference matters. |
| **Require pgvector >= 0.8 and refuse to run on older versions** | Simpler code, but hostile to developers with an older local image and to any environment where the extension version is not under our control. The fallback path is small and its logs make it visible. |
| **Config knob for the over-fetch multiplier** | Moves a decision that has a correct answer (enough overscan to satisfy the contract on realistic filters) into runtime configuration. A config knob invites drift between environments and makes evaluation reports harder to compare. Constants in code, changed deliberately, are better. |
| **Automatic upgrade of the extension at startup** | `CREATE EXTENSION` / `ALTER EXTENSION UPDATE` requires elevated privileges. A service should not modify the database schema on boot. Migrations are the place for that, and pgvector upgrades are not something the service does implicitly. |
| **Detect version per request** | Wastes a round-trip and adds noise to logs. The version is stable for the life of a connection. |

## Consequences

**Positive**

- **The contract is honored on modern pgvector.** A filtered query returns
  `k` results whenever the catalog has `k` matching items, which is what the
  API doc promises.
- **Behavior differences are visible.** A query on pgvector 0.7 and a query
  on pgvector 0.8 both produce log lines that identify which path they took.
  An operator comparing evaluation reports sees the version in the report
  metadata and knows not to mix them.
- **The filter allowlist has one source.** `contracts.md` § 2.1 is
  authoritative; the code references it via a single constant. A field added
  to the contract is added in one place in the code.
- **No string-built SQL.** Every filter value is a parameter. The rule is
  mechanical: if a value from a request appears in a query, it is `%s` or
  `%(name)s`, never an f-string.

**Negative / accepted trade-offs**

- **`strict_order` is slower than `relaxed_order` on very selective
  filters.** The difference is measured in M2.4. If it turns out to be
  material on the sample catalog, switching is one line and an update to the
  design doc; the ADR will be superseded, not edited.
- **The `k * 4` fallback is a heuristic.** It is a reasonable starting
  point for filters that are moderately selective (say, 10–30% selectivity),
  but it is not a general solution. On pgvector 0.7, a filter matching 1% of
  a 1M-item catalog would need `k * 100` at least, which exceeds any sane
  multiplier. The correct answer for such filters is to upgrade pgvector,
  or to add a partial index (per `docs/adr/0001-pgvector-as-default.md`
  § Alternatives), or to use a two-stage strategy: retrieve from a filtered
  subset with a partial index. The fallback path is documented as
  "correct for typical selectivity, inadequate for extreme selectivity"; a
  future ADR will address partial indexes if the project ever needs them.
- **The version is captured at process start.** A live upgrade of the
  extension on a running database is not picked up until the service is
  restarted. Acceptable: extension upgrades in production are coordinated
  maintenance, not a hot swap.
- **`max_scan_tuples` is a magic number.** Bounded because a query must
  terminate, but the value is not derived from first principles. M2.4 will
  publish the measured relationship between filter selectivity and needed
  scan size on the sample catalog; the constant is revisited then.

## References

- `docs/adr/0001-pgvector-as-default.md` — why pgvector and HNSW
- `docs/adr/0006-pgvector-schema.md` — the schema whose filters this ADR
  serves
- `docs/adr/0007-index-version-identity.md` — how `pgvector_version` enters
  `index_version`
- `docs/contracts.md` § 2.1 — the filter allowlist and the response
  ordering guarantee
- `docs/retrieval-and-evaluation.md` — the M2 design doc, which records the
  measured `strict_order` vs `relaxed_order` trade-off
- pgvector documentation, `hnsw.iterative_scan` and `hnsw.max_scan_tuples`
