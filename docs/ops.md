# Operations

Operational procedures for the running system. The runbook in
[`runbook.md`](runbook.md) covers incident response (alerts, symptoms,
diagnosis). This document covers **planned operations**: backups, restores,
index lifecycle, disk-space planning, and the procedures that are performed
on purpose rather than in response to a failure.

> **Status:** M2 slice. Sections marked *(planned)* land in later milestones;
> they are listed here so the eventual procedures have a named home, and so
> the boundary between "defined" and "not yet defined" is visible.

---

## 1. What exists

| Component | State | Where |
|---|---|---|
| `item`, `embedding`, `index_registry` tables | M2 | PostgreSQL 16 + pgvector |
| Embedding artifacts (Parquet runs) | M1 | `artifacts/embeddings/runs/<run_id>/` |
| ONNX model artifact | M1 | `artifacts/onnx/<model_slug>/` |
| Redis cache | *(M3)* | Redis 7 |
| Event log | *(M5)* | PostgreSQL |
| Churn model registry | *(M7)* | PostgreSQL |

The operational surface of M2 is therefore:

- The PostgreSQL database (schema + index + registry).
- The embedding artifact tree on disk.
- The ONNX artifact on disk.

Everything else is either unchanged from M1 or still planned.

---

## 2. Backup

### 2.1 What to back up

**PostgreSQL — logical backup, per-database:**

```bash
pg_dump --format=custom --no-owner --no-privileges \
    --file=recsys-$(date -u +%Y%m%dT%H%M%SZ).dump \
    "$DATABASE_URL"
```

Covers `item`, `embedding`, `index_registry`, and the alembic version
table. `--format=custom` supports parallel restore and selective restore
(`pg_restore --table=...`), which is what a partial-recovery scenario
needs.

**Embedding artifact tree — filesystem copy:**

```bash
tar -C artifacts/embeddings -czf embeddings-$(date -u +%Y%m%dT%H%M%SZ).tar.gz .
```

The tree contains the Parquet run files, the `manifest.json` and
`state.json` per run, and the `current` pointer. It does **not** contain
the ONNX artifact.

**ONNX artifact — filesystem copy:**

```bash
tar -C artifacts/onnx -czf onnx-$(date -u +%Y%m%dT%H%M%SZ).tar.gz .
```

Reproducible from source (see § 4.1), but backing it up avoids a
re-download and re-export on restore.

### 2.2 When to back up

- **Before** a model swap (ADR-0006 § Model swap procedure).
- **Before** a pgvector extension upgrade.
- **Before** any Alembic migration that alters a table.
- **After** a successful `make index-promote` (so the new active state is
  captured), unless a scheduled backup is imminent.
- **Scheduled:** daily logical backup, weekly artifact backup. The
  schedule is a deployment concern, not a code concern; the commands above
  are what the scheduler runs.

### 2.3 What a backup does not cover

- **Redis contents.** The cache is reconstructible. On restore, it starts
  cold. This is by design (`docs/decisions.md` § 4: the cache is optional
  on the hot path).
- **In-flight experiment state** *(M5)*. Experiment assignments are
  recomputed from the user id and the salt; a backup that misses a day of
  exposure events loses that day's measurements, but not the experiment's
  ability to continue.
- **The pgvector extension itself.** `pg_dump` includes `CREATE EXTENSION
  vector;` in the output only if the extension is not owned by a template
  database. On restore, the extension must be present in the target
  cluster; the restore command that enables it is in § 3.2.

### 2.4 Retention

Logical backups: 30 days. Artifact backups: 30 days, plus the most recent
backup from each of the last 3 months. The retention policy is
deliberately short: the artifacts are reproducible from the catalog and
the ONNX model (§ 4.1), so a backup is a convenience, not the last line
of defense.

---

## 3. Restore

### 3.1 Scenario: the embedding table is corrupt or was dropped

1. Stop the API (drain, then stop). In M2 the API does not yet serve
   retrieval, so this step is a no-op; from M3 onward it is required.
2. Restore the database:

   ```bash
   pg_restore --clean --if-exists --no-owner --no-privileges \
       --dbname="$DATABASE_URL" recsys-<timestamp>.dump
   ```

3. Verify the active index is present and its `row_count` matches:

   ```bash
   psql "$DATABASE_URL" -c "
     SELECT index_version, status, row_count
     FROM index_registry
     WHERE status = 'active';
   "
   psql "$DATABASE_URL" -c "SELECT count(*) FROM embedding;"
   ```

   The `row_count` from the registry and the count from `embedding` must
   agree. If they do not, the restore was partial — re-run it, do not
   proceed.

4. Start the API. `/readyz` must report ready (M3).

### 3.2 Scenario: the cluster is empty (new machine)

1. Provision PostgreSQL 16 with the pgvector extension available. The
   image `pgvector/pgvector:pg16` satisfies this.

2. Create the database and the extension:

   ```bash
   createdb recsys
   psql "$DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. Restore from the most recent logical backup (§ 2.1).

4. Run any migrations newer than the backup:

   ```bash
   alembic upgrade head
   ```

5. Restore the artifact tree (§ 2.1) to `artifacts/embeddings/`. The
   `current` pointer is part of the tree; do not reconstruct it by hand.

6. Verify `index_registry` and `embedding` agree (§ 3.1, step 3).

### 3.3 Scenario: an embedding run directory is corrupt

The artifact tree is append-only; a single run directory can be removed
without affecting others. If the **active** run is corrupt:

1. Point `current` at the previous run. The pointer is a one-line text
   file:

   ```bash
   ls artifacts/embeddings/runs/            # list candidate run_ids
   printf '%s\n' '<previous_run_id>' > artifacts/embeddings/current
   ```

   No restart of the pipeline is required; consumers read `current` on
   demand. The API (M3) caches the active index at startup, so this step
   is followed by a drain-and-restart of API instances.

2. Rebuild the corrupt run from source if needed:

   ```bash
   python -m recsys.embeddings.pipeline --mode=batch \
       --catalog data/sample/catalog.jsonl \
       --out artifacts/embeddings
   ```

   The rebuild produces a new `run_id`; the corrupt directory can be
   removed afterward.

---

## 4. Rebuilding from source

### 4.1 Rebuilding the ONNX artifact

The ONNX artifact is derived from the model name and the export script.
The `model_version` in every manifest records the SHA of the artifact that
produced the embeddings, so a rebuild that produces a different SHA
produces a different `model_version`, and by ADR-0007 a different
`index_version`.

```bash
python scripts/export_onnx.py \
    --model sentence-transformers/all-MiniLM-L6-v2
```

The script is idempotent: if the artifact exists and its SHA matches the
sidecar, it is skipped. `--force` re-exports.

### 4.2 Rebuilding an index from an embedding run

```bash
python scripts/build_index.py --run-id <run_id>
python scripts/build_index.py --promote
```

`build_index.py` reads the Parquet for `<run_id>`, computes the target
`index_version` (ADR-0007), and if a row with identical inputs already
exists, exits without writing. Idempotent by construction.

### 4.3 Full rebuild (nuclear option)

When the artifact tree and the database are both gone and only the source
tree remains:

1. `make install-dev` to install the extras.
2. `python scripts/export_onnx.py --model <model>`.
3. `python -m recsys.embeddings.pipeline --mode=batch --catalog
   data/sample/catalog.jsonl --out artifacts/embeddings`.
4. `alembic upgrade head`.
5. `python scripts/build_index.py --promote`.
6. `make eval` to confirm the thresholds still pass.

Steps 2–6 run to completion in under a minute on the sample catalog. On a
1M-item catalog, step 3 dominates and takes minutes to tens of minutes
depending on CPU; step 5 depends on the embedding insert throughput.

---

## 5. Index lifecycle operations

### 5.1 Build, promote, rollback

```bash
make index-build                    # build from the active run, status=building
make index-promote VERSION=idx-a3f9e021
make index-rollback                 # active → retired, previous retired → active
```

Full lifecycle and state machine in
[`retrieval-and-evaluation.md`](retrieval-and-evaluation.md) § 11.

### 5.2 Advisory lock during promote

`scripts/promote_index.py` takes a `pg_advisory_lock` on a fixed key
before it changes any registry row:

```sql
SELECT pg_advisory_lock(hashtext('recsys.index_promote'));
-- ... updates ...
SELECT pg_advisory_unlock(hashtext('recsys.index_promote'));
```

The lock is **transaction-scoped by convention**, not by the database:
`pg_advisory_lock` is session-scoped, and the script holds it for the
duration of the promote transaction, then releases it explicitly. If the
process crashes mid-promote, the lock is released when the connection
closes and the promote is not partially applied — the transaction rolls
back.

**Why a separate lock from the transaction.** Two concurrent promotes are
rare but not impossible (a scheduled promote and a manual one). Without
the advisory lock, both transactions would try to retire the same active
row and activate different targets. The partial unique index on
`status = 'active'` would reject the second, but as an opaque constraint
violation. The advisory lock turns the conflict into a serialization:
the second promote waits, then sees that the target it intended to
promote is already active and exits cleanly.

**Why not `SELECT ... FOR UPDATE` on the registry row.** The row being
updated is the target (in `building` status) and the previous active row.
Locking both requires two locks in a defined order; a promote and a
rollback racing on the same pair could deadlock. A single advisory lock
has no ordering problem.

### 5.3 Rollback is a pointer swap

A rollback changes exactly two registry rows: the current active becomes
`retired`, the most recent retired becomes `active`. The `embedding` rows
for both indexes are still present; the rollback does not move data. This
is why rollback is fast (sub-second) and why blue/green is worth the
disk cost.

### 5.4 Re-promoting a retired index

`make index-promote VERSION=<v>` refuses to promote a row that is
`retired`. If an operator wants to reactivate a previously retired index,
the correct operation is `make index-rollback`, which picks the most
recent retired row. Directly promoting an older retired row is not
supported; if it becomes necessary, the promote script is extended with a
`--force` flag and the reason is documented in the commit. There is no
current use case.

---

## 6. Disk-space planning

### 6.1 PostgreSQL

Per index version:

```
bytes ≈ row_count × (384 × 4 + overhead)
      ≈ row_count × 1600   # empirical, including the HNSW graph and page overhead
```

| Catalog size | Per index | Two indexes (blue/green) |
|---|---|---|
| 200 items (sample) | ~320 KB | ~640 KB |
| 100k items | ~160 MB | ~320 MB |
| 1M items | ~1.6 GB | ~3.2 GB |

The HNSW graph is the dominant term for large catalogs; the vectors
themselves are 1.5 KB per row at 384 dims float32.

### 6.2 Embedding artifact tree

Per run:

```
bytes ≈ row_count × (384 × 4 + len(content_hash) + overhead)
      ≈ row_count × 1600   # Parquet with zstd compression, empirical
```

The sample catalog is ~120 KB per run. A 1M-item catalog is ~1.6 GB per
run. The tree grows with every batch run and every incremental run that
produces a new snapshot; incremental runs whose inputs are unchanged
produce no new run.

### 6.3 When to prune

- **Postgres embeddings:** when the sum of all `retired` index versions
  exceeds a configured threshold (default: 2× the size of the active
  index). A future `make prune-indexes --keep=N` target will delete
  retired rows whose `retired_at` is older than N days and that are not
  the most recent retired row (which is the rollback target). Not
  implemented in M2.
- **Artifact tree:** when the tree exceeds a configured threshold
  (default: 5 GB) or when the number of runs exceeds a configured count
  (default: 20). A future `make prune-embeddings --keep=N` target will
  remove the oldest runs, always preserving the run named by `current`
  and the run named by `state.json` of the active run. Not implemented in
  M2.

Both prune operations are deferred, not forgotten. The thresholds are
numbers, not "later".

---

## 7. Failure modes specific to M2

These complement `runbook.md`, which covers the general service failure
modes. The entries below are the ones that only appear once the retrieval
layer is live.

### 7.1 `make index-build` fails midway

**Symptom:** the command exits non-zero; `index_registry` has a row in
`building` status with `row_count = 0` or a partial value.

**Diagnose:** the log line `index.build.error` names the failing step. The
most common causes are a missing Parquet run (the `current` pointer names
a run that was pruned) and a database constraint violation (a mismatched
`item_id` between the Parquet and the `item` table).

**Mitigate:** the `building` row is invisible to readers; no consumer is
affected. Delete the row and re-run `make index-build`. If the Parquet is
the problem, restore the artifact tree (§ 3.3) and retry.

### 7.2 Two promotes race

**Symptom:** one promote succeeds, the other exits with a lock timeout
or sees the target already active. Log line `index.promote.done` appears
once.

**Diagnose:** expected behavior when two operators trigger a promote in
the same second. The advisory lock (§ 5.2) serializes them.

**Mitigate:** none required. The second promote's target is already
active; the operation is idempotent at the outcome level. If the second
promote had intended a different target, the lock timeout is the signal
to retry.

### 7.3 Filtered query returns fewer than `k` results

**Symptom:** a request with a selective filter returns fewer items than
requested, with no error.

**Diagnose:** this is either the intended behavior (§ 2.1 of
`contracts.md`: "may be smaller when filters are very selective") or the
result of the `hnsw.max_scan_tuples` bound on pgvector 0.8+. Check the
detected version and the query plan:

```bash
psql "$DATABASE_URL" -c "
  SET hnsw.iterative_scan = 'strict_order';
  EXPLAIN (ANALYZE, FORMAT JSON)
  SELECT item_id FROM embedding
  WHERE index_version = 'idx-...'
    AND item_id IN (SELECT item_id FROM item WHERE category = 'rare_category')
  ORDER BY vector <=> '[...]'
  LIMIT 10;
"
```

If the plan shows `max_scan_tuples` reached, the filter is more selective
than the current bound accommodates. The remediation is either to raise
the bound (a code change, with a commit that records the reason) or to
address the underlying selectivity (a partial index, deferred; see
ADR-0008).

### 7.4 pgvector version mismatch between CI and production

**Symptom:** the evaluation gate passes in CI (pgvector 0.8+) but a
production environment logs `retrieval.pgvector.old_version` at startup
and `retrieval.pgvector.fallback` on filtered queries.

**Diagnose:** the production environment has not been upgraded. The
registry row for the active index records the version that built it; if
it differs from the runtime version, the environment is running an index
built by a different extension release. This is not itself an error — the
index is valid — but the runtime behavior differs.

**Mitigate:** upgrade the pgvector extension in the production
environment. Coordinate with § 2.2 (backup before upgrade) and § 4.2
(rebuild the index after upgrade if the upgrade affects recall).

---

## 8. What is planned but not defined

The following procedures have no written runbook yet. They are listed so
that the gap is explicit:

| Operation | Lands in |
|---|---|
| Redis cache invalidation and warmup | M3 |
| Rate-limit bucket inspection | M3 |
| Experiment SRM investigation | M5 |
| Deployment rollback (code) | M6 |
| Staging→production promotion | M6 |
| Churn feature-store repair | M7 |
| Trivy scan remediation | M8 |

Until each is defined, the general fallback is: drain the affected
instances, roll back to the last known-good artifact set (M1) or index
version (M2), and file an issue.

---

## 9. References

- [`runbook.md`](runbook.md) — incident response (alerts, symptoms,
  diagnosis, escalation)
- [`retrieval-and-evaluation.md`](retrieval-and-evaluation.md) § 11 —
  index lifecycle and state machine
- [`adr/0006-pgvector-schema.md`](adr/0006-pgvector-schema.md) — schema and
  model-swap procedure
- [`adr/0007-index-version-identity.md`](adr/0007-index-version-identity.md)
  — how `index_version` is computed and where it is stored
- [`adr/0008-filter-strategy.md`](adr/0008-filter-strategy.md) — filter
  behavior and the pgvector version fallback
- [`embedding-pipeline.md`](embedding-pipeline.md) § 7 — artifact tree
  layout and the `current` pointer
