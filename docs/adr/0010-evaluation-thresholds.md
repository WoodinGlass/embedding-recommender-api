# ADR-0010: Evaluation thresholds — absolute floor, history, and the CI gate

- **Status:** Accepted
- **Date:** 2026-10-08
- **Deciders:** project maintainer

## Context

M2 ships an offline evaluation that computes Recall@k, NDCG@k, MRR, and ANN
fidelity (see ADR-0009). The exit criterion is not "the numbers are printed"
— it is "the numbers are enforced as a CI gate". That requires a definition
of *how low is too low*, and a definition of what happens when a run falls
below that line.

Three failure modes are possible in any threshold-based gate:

1. **The threshold is too generous.** A regression that halves retrieval
   quality still passes, because the threshold was set slightly below the
   initial measurement. The gate reports green and nothing has been caught.
2. **The threshold is too strict.** A small, acceptable fluctuation in a
   metric — caused by an upstream dependency update, a different CPU, or an
   intentionally-changed but still-good parameter — fails the build. The
   team learns to ignore the gate, and it stops catching real regressions.
3. **The threshold drifts silently.** Someone lowers the threshold to make a
   failing build pass. There is no record of the change, no reason given,
   and no way for a future reviewer to know that the gate was weakened.

Each of these is a design decision, not a tuning problem. The threshold file
must be designed so that all three failure modes are visibly hard.

## Decision

### Threshold structure

`evaluation/thresholds.yaml`, versioned in git:

```yaml
schema_version: 1
golden_set_version: "v1"

# Absolute floors: a run that falls below any floor fails the gate,
# regardless of how the threshold was chosen. These encode the minimum
# quality this project considers acceptable for the sample catalog.
absolute_floor:
  pgvector_hnsw:
    recall_at_10: 0.70
    ndcg_at_10: 0.65
    mrr: 0.70
    ann_recall_vs_exact: 0.90

# Per-system thresholds: a run below any of these fails the gate. Values
# are chosen after reviewing a measured baseline; see "How thresholds are
# set" below.
pgvector_hnsw:
  recall_at_10: 0.85
  ndcg_at_10: 0.80
  mrr: 0.85
  ann_recall_vs_exact: 0.90

exact_knn:
  recall_at_10: 0.85
  ndcg_at_10: 0.80
  mrr: 0.85
  ann_recall_vs_exact: 1.00

popularity_synthetic:
  recall_at_10: 0.00
  ndcg_at_10: 0.00
  mrr: 0.00
  ann_recall_vs_exact: null

random_baseline:
  recall_at_10: 0.00
  ndcg_at_10: 0.00
  mrr: 0.00
  ann_recall_vs_exact: null
```

**Schema version** allows the file format to change without breaking the
gate: a mismatch is a fail, not a silent pass.

**`golden_set_version`** ties the thresholds to the golden set they were
chosen for. A report whose golden set version does not match this field is
a fail (see "Staleness and version checks" below).

**`absolute_floor`** is a per-metric lower bound that applies *in addition
to* the per-system threshold. The gate checks both. This is the answer to
failure mode (1): it makes it impossible to lower a per-system threshold
into territory that is objectively unacceptable.

**Baselines have `0.00` thresholds.** The point of a baseline is to be a
floor. If the random baseline somehow fails to score 0.00 on recall — for
example because the "random" seed produced a deterministic ranking that
coincidentally matched relevant items — that is a bug in the baseline, and
failing the gate is the correct response.

### How thresholds are set

Thresholds are **not** derived automatically from the measured value. They
are chosen by a human, in this order:

1. Run `make eval` once, with no threshold file (or with permissive
   thresholds). Read the measured values.
2. Compare against the absolute floors. If any metric is below the floor,
   **stop**. A system that does not meet the minimum should not have its
   threshold pinned above that floor and shipped. Either fix the system or
   lower the floor with a written reason (see "Threshold history" below).
3. If all metrics are above the floor, set the per-system threshold to the
   measured value minus a margin. The margin is per-metric and per-metric
   realistic:
   - Metrics that are mathematically stable (recall@10 with a fixed golden
     set and a pinned index) get a small margin: 0.02–0.05.
   - Metrics that depend on ANN behavior (ANN fidelity) can fluctuate more
     with `ef_search` and with hardware, so the margin is larger: 0.05.
4. Commit the threshold file with a message that says which commit the
   measurement came from and what the measured values were.

**Why not `threshold = measured - margin` blindly.** A measured value can
itself be a regression. If a change drops NDCG@10 from 0.85 to 0.60 without
anyone noticing, and the threshold is automatically recomputed to 0.55, the
regression is now legal and the gate will never catch a future 0.54. The
absolute floor is the answer: 0.65 in the example above would have failed
the run immediately.

**Why a human sets the threshold.** The automation that would choose the
threshold cannot distinguish "the metric dropped because the code changed
correctly and the previous number was provisional" from "the metric dropped
because something is wrong". A human who reads the diff can. This is the
same reason the ADR process is not automated.

### Staleness and version checks

The gate compares three things before it evaluates any threshold:

1. **Golden set version.** `thresholds.yaml` declares a version.
   `report.json` records the version actually used. If they differ, the
   gate fails with `event="eval.golden_set_version.mismatch"` and does not
   evaluate metrics. A threshold for one golden set is not valid for
   another.
2. **Report commit.** `report.json` records the git commit it was produced
   from (via `git rev-parse HEAD` at evaluation time). The gate compares
   this against the commit under test. If they differ, the gate fails with
   `event="eval.report.stale"`. A report from an older commit is not
   evidence about the current one.
3. **Report age.** `report.json` records a wall-clock timestamp. The gate
   fails if the report is older than a configurable `max_report_age_hours`
   (default 24). This catches the case where a report was produced and then
   cached across many CI runs of different code.

If any check fails, the gate **fails**, not skips. A silently skipped gate is
the same as a disabled gate.

### Threshold history

`evaluation/thresholds_history.yaml`, appended whenever
`thresholds.yaml` changes:

```yaml
schema_version: 1
entries:
  - date: "2026-10-08T12:00:00Z"
    golden_set_version: "v1"
    commit: "<sha>"
    change: "initial thresholds"
    measured:
      pgvector_hnsw:
        recall_at_10: 0.91
        ndcg_at_10: 0.87
        mrr: 0.90
        ann_recall_vs_exact: 0.97
    new_thresholds:
      pgvector_hnsw:
        recall_at_10: 0.88
        ndcg_at_10: 0.85
        mrr: 0.88
        ann_recall_vs_exact: 0.92
    reason: null
  - date: "2026-11-15T10:00:00Z"
    golden_set_version: "v1"
    commit: "<sha>"
    change: "lower recall_at_10 for pgvector_hnsw"
    measured:
      pgvector_hnsw:
        recall_at_10: 0.83
    new_thresholds:
      pgvector_hnsw:
        recall_at_10: 0.80
    reason: "HNSW ef_search lowered from 100 to 60 to meet the p95 latency target. Recall cost is measured and accepted; the ADR for the latency/recall trade-off is ADR-NNNN."
```

The history file is not decorative. It exists so that:

- A future reviewer can see when a threshold was lowered and why.
- A threshold that is lowered **without** a written reason is visible as a
  diff anomaly. The gate itself does not enforce the presence of a reason —
  that is a matter for code review — but the file makes the absence obvious.
- The measured values at the time of a threshold change are preserved, so
  the relationship between measurement and threshold is not lost.

**Rule:** lowering a threshold requires a `reason` field. Raising or
re-computing a threshold after a golden-set version bump does not.

### CI job

A dedicated CI job, separate from `test-integration` and `test-encoder`:

```yaml
evaluation:
  name: evaluation gate
  runs-on: ubuntu-latest
  timeout-minutes: 10
  needs: [lint]
  services:
    postgres:
      image: pgvector/pgvector:pg16
      # ... same service config as test-integration
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: pip
        cache-dependency-path: pyproject.toml
    - name: Install
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev-lite,db,pipeline,inference,export]"
    - name: Prepare pgvector extension
      run: |
        sudo apt-get update && sudo apt-get install -y --no-install-recommends postgresql-client
        PGPASSWORD=recsys psql -h localhost -U recsys -d recsys -c "CREATE EXTENSION IF NOT EXISTS vector;"
    - name: Produce ONNX artifact
      run: python scripts/export_onnx.py --model sentence-transformers/all-MiniLM-L6-v2
    - name: Embed the sample catalog
      run: python -m recsys.embeddings.pipeline --mode=batch --catalog data/sample/catalog.jsonl --out artifacts/embeddings --quiet
    - name: Apply migrations
      run: alembic upgrade head
    - name: Build and promote the index
      run: |
        python scripts/build_index.py --promote
    - name: Run evaluation
      run: make eval
```

**Why a separate job.** The gate is a distinct failure mode from the
integration tests. If `evaluation` fails and `test-integration` also fails,
the two failures must be distinguishable. Mixing them into one job would
require reading the log to know which phase failed, and would let one
mask the other if `set -e` were ever misused.

**Why `timeout-minutes: 10`.** The golden set is 20 queries; on the sample
catalog (200 items), an evaluation pass runs in seconds after setup. The
timeout exists to catch runaway queries, not to accommodate slow runs. If
the golden set grows, the timeout grows with it, and the change is
deliberate.

**Why `needs: [lint]` and not `needs: [test-integration]`.** The lint job
catches syntax and type errors in seconds. The evaluation job depends on
those being clean, but it does not depend on the integration tests passing.
They exercise different code paths and can run in parallel.

### `make eval` interface

The command:

```
make eval
```

runs `python -m recsys.evaluation.runner --thresholds evaluation/thresholds.yaml --report evaluation/report.json`.

- Exit code 0: all systems meet all thresholds (per-system and absolute
  floor), golden set version matches, report is fresh.
- Exit code 1: one or more thresholds not met, or a version/staleness check
  failed. The failing metrics are printed to stderr in JSON lines and
  summarized on stdout in a single JSON line.
- Exit code 2: could not run evaluation (missing index, missing golden set,
  database unreachable). This is distinct from "ran and failed" — the two
  cases have different remediation.

The report file (`evaluation/report.json`) is written on exit code 0 or 1,
not on exit code 2 (there is nothing to report if evaluation did not run).
CI uploads the report as a build artifact regardless of exit code, so a
failure can be inspected after the fact.

## Alternatives considered

| Option | Why not |
|---|---|
| **`threshold = measured - margin` only, no absolute floor** | Failure mode (1): a regression that drops the metric across many runs is laundered into a lower threshold, and the gate converges to "whatever the code happens to do today". The absolute floor is the minimum cost of an honest gate. |
| **Threshold as a ratio of the previous measured value (e.g. `>= 0.95 * last`)** | Requires storing the last measured value and updating it on each run. The gate would drift: a series of 5% drops compounds to a 22% total drop over five runs, each of which individually passes. Absolute thresholds do not drift. |
| **Per-run thresholds computed from a baseline file that changes rarely** | Same as above, plus an extra file to keep in sync. The absolute floor plus a manually-chosen per-system threshold is simpler and has the same effect. |
| **Skip the gate when the golden set version changes** | A silently skipped gate is a disabled gate. If the golden set changed, the thresholds must change too, and the run that changes the golden set is the run that updates the thresholds. Fail-on-mismatch forces the coupling to be visible. |
| **Auto-generate thresholds on the first run and commit them** | The first run is not necessarily the right run. If the first run is on a bad commit, the thresholds are calibrated to a bad baseline, and the absolute floor is the only thing that catches it — which is why the floor exists even with auto-generation. Manual calibration is one review cycle more expensive and one class of failure safer. |
| **No history file** | Failure mode (3): threshold drift is silent. The history file does not enforce honesty, but it makes dishonesty visible. For a portfolio project whose point includes reviewable decisions, that visibility is worth the extra file. |
| **Gate on the diff between the current report and the last committed report** | Requires the last report to be committed, which means either committing report.json (churn) or storing it as a CI artifact with an out-of-band reference. Also conflates "regression relative to last run" with "regression relative to an acceptable minimum". Thresholds and diffs answer different questions; this project answers the threshold question first. |
| **Run evaluation inside `test-integration`** | Conflates two failure modes. A database setup failure and a retrieval quality failure would both show up as "test-integration failed", and the correct response to each is different. |
| **Timeout of 30 minutes to "be safe"** | A timeout that never fires is a timeout that does not do its job. Ten minutes is longer than any legitimate evaluation of the sample catalog by two orders of magnitude; it exists to catch a hung query, not a slow one. |

## Consequences

**Positive**

- **Failure mode (1) is closed.** The absolute floor prevents a slow drift
  from laundering regressions into the thresholds. A system that drops below
  the floor fails the gate regardless of what the per-system thresholds say.
- **Failure mode (2) is bounded.** Per-system thresholds have margins
  calibrated to what each metric does naturally. Baselines have thresholds
  of `0.00` because that is what they should score — there is no upward
  fluctuation to accommodate.
- **Failure mode (3) is visible.** The history file records every threshold
  change, with the measured values at the time and, when applicable, a
  reason. A dropped threshold without a reason is a diff anomaly, not a
  silent edit.
- **The gate fails on mismatch, not skip.** A stale report, a mismatched
  golden set version, or an unreachable database all produce failures, not
  green runs. This is the only setting in which the gate is trustworthy.
- **Exit codes distinguish "failed" from "could not run".** The two have
  different remediation; conflating them wastes operator time.

**Negative / accepted trade-offs**

- **Threshold calibration is manual.** One more review step for every
  significant change. Accepted: the alternative (automated calibration) does
  not distinguish "the metric moved because the code is correct" from "the
  metric moved because the code is wrong".
- **The history file can grow long.** For a project at this scale, an entry
  per threshold change is at most a few dozen lines. If it ever becomes
  burdensome, a compaction pass that keeps the last N entries plus any
  entries with a reason can be added — but not before it is a problem.
- **Absolute floors are arbitrary at the start.** `recall_at_10: 0.70` is
  not derived from any theory; it is what the maintainer considers the
  minimum acceptable for a working sentence-embedding model on the sample
  catalog. If measurements later show that this is either trivially easy
  or impossible for a legitimate system, the floor is revised (with a
  history entry and, if lowered, a reason). This is a starting point, not
  a derivation.
- **Report freshness depends on CI running once per commit.** If two commits
  are pushed in quick succession and CI cancels the first run, the second
  run produces a report for the second commit, and the first commit's report
  never existed. This is fine — the first commit is superseded. But if
  branch protection ever requires a report for every commit individually,
  the freshness check would need to accommodate it. Not a M2 concern.
- **The gate does not check for regressions relative to the last commit.**
  It checks against thresholds. A 10% drop that stays above the threshold
  passes. This is deliberate: the threshold is the minimum acceptable, and
  a drop above it is legitimate. If the project wants per-commit regression
  detection, that is a separate mechanism (a diff-based check) and a
  separate ADR.

## References

- `docs/adr/0009-golden-set-and-metrics.md` — metric definitions the
  thresholds apply to
- `docs/retrieval-and-evaluation.md` — the M2 design doc
- `evaluation/thresholds.yaml` — the file this ADR specifies
- `evaluation/thresholds_history.yaml` — the audit trail
- `evaluation/report.json` — the per-run output the gate reads
- `Makefile` — the `eval` target
- `.github/workflows/ci.yml` — the `evaluation` job
