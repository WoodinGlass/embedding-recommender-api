# ADR-0004: Versioned run directories with an atomic `current` pointer

- **Status:** Accepted
- **Date:** 2026-10-07
- **Deciders:** project maintainer

## Context

The first draft of M1.3 wrote each artifact set directly under
`artifacts/embeddings/<model_version>/<catalog_snapshot>.parquet` plus two
sidecars (`manifest.json`, `checksums.json`) and an incremental `state.json`
alongside. A review identified four problems with that layout:

1. **No atomic commit.** The pipeline wrote Parquet, then manifest, then
   state, in that order. A crash between any two writes left the directory
   in an inconsistent state: an artifact with no manifest, or a state file
   pointing at a Parquet that does not exist.
2. **No single entry point.** Consumers (M2 retrieval, the M2 eval gate, M3
   API) had to reconstruct the expected filename from `model_version` and
   `catalog_snapshot`, or scan the directory for the newest file by mtime.
   Both are fragile.
3. **No rollback.** Rolling back to the previous embedding set meant renaming
   files by hand or re-running the pipeline — there was no way to say "use
   the artifact from the previous run" without knowing the previous run's
   snapshot identifier.
4. **No isolation between runs.** Two pipeline invocations targeting the same
   `model_version` would overwrite each other's Parquet if they ran
   concurrently, with no lock and no detection.

## Decision

Adopt a **versioned-run layout with a single atomic file pointer**:

```
artifacts/embeddings/
├── runs/
│   └── <run_id>/
│       ├── embeddings.parquet
│       ├── manifest.json
│       └── state.json
├── current                      # one-line text file, path relative to runs/
└── .lock                        # fcntl.flock file (empty; PID/host written on acquire)
```

`<run_id>` is `<ISO8601-Z>__<model_version>`, e.g.
`2026-10-07T11-30-00Z__minilm-onnx-v1+a3f9e021`. The format is documented in
`docs/contracts.md` § 1.3.

The commit protocol is fixed and executed in this order:

1. Write `embeddings.parquet` into a temp file inside the run directory,
   `fsync`, `os.replace` onto the final name.
2. Write `manifest.json` the same way.
3. Write `state.json` the same way.
4. Write `current.tmp` containing the run_id + newline, `fsync`, `os.replace`
   onto `current`.

Step 4 is what makes the run *visible*. Anything before it is a candidate
that can be discarded; after it, the run is live. A crash anywhere in steps
1–3 leaves the previous `current` in place, so the system continues to serve
the previous artifact.

`current` is a **file pointer (one line of text), not a symlink.** Symlinks
were rejected because they are not reliably atomic on NFS, EFS, and some
Kubernetes CSI drivers, and they have no equivalent on object storage. A file
pointer is portable, debuggable (`cat current`), and migratable to any
storage backend that supports a rename.

A **`fcntl.flock` exclusive lock** on `.lock` serialises pipeline runs.
`flock` releases automatically when the process dies (kernel-managed), so
there is no stale-lock problem. NFS is documented as unsupported; a future
move to NFS or an object store will require a different locking strategy (a
new ADR).

## Alternatives considered

| Option | Why not |
|---|---|
| **Keep the flat layout, add a lock and a pointer** | Solves items 1 and 4 but leaves 2 and 3: consumers still reconstruct filenames, and rollback is still manual. The pointer helps only if it points at a self-contained run. |
| **Symlink `current` instead of a text pointer** | Atomic on Linux local filesystems, but not reliably on NFS, EFS, or many CSI drivers; no equivalent on S3 or GCS. Migrating to object storage later would force a second change. The file pointer costs one extra read and works everywhere. |
| **No pointer; consumers scan for newest mtime** | mtime is fragile: `git checkout`, `cp -p`, backups, and rsync can all change it without touching content. Also hides corruption (a "newest" run could be a partial write). |
| **SQLite manifest / database** | Adds a dependency and a second state store for something the filesystem already models. The pipeline targets single-node local disk; a database is overkill. |
| **Symlink tree `runs/current/ -> <run_id>/`** | Same portability problems as a symlink to a file, with the added risk of dangling links if a run directory is partially deleted. |

## Consequences

**Positive**

- **Atomic commit.** A consumer that reads `current` and then follows the
  pointer sees either the previous run (before step 4) or the new run (after
  step 4). Never an intermediate state.
- **Rollback is one line.** Write the previous `run_id` into `current`. No
  file moves, no re-encode.
- **Single entry point for consumers.** The M2 retrieval loader, the eval
  gate, and the M3 API all read `current`. None of them needs to know the
  run_id format or the artifact filenames.
- **Locking is stdlib.** `fcntl.flock` requires no extra dependency and
  releases on process death. The current target (Linux local disk: Colab,
  GitHub Actions, Docker) is exactly where `flock` is reliable.
- **Portable.** A file pointer works on local disk, NFS, and (with a small
  change to the pointer resolution) object storage. The pointer's content is
  a relative path, so the artifact directory can be moved or copied without
  rewriting it.

**Negative / accepted trade-offs**

- **One extra read** per consumer startup: `current` then `manifest.json`.
  Measured in microseconds; not worth optimising.
- **Old runs accumulate on disk.** There is no automatic garbage collection.
  A future ADR will add a `make prune-embeddings --keep=N` target. For the
  sample catalog and the M2 workload, accumulation is negligible.
- **`flock` is not reliable on NFS.** Documented in the runbook and here.
  Moving to NFS, EFS, or an object store will require either `filelock` with
  stale-lock detection or an external lock service. This is an explicit
  future ADR, not a silent gap.
- **`.lock` file remains after a clean run.** It is an empty file by design;
  the lock is held by the file descriptor, not the file's existence. This
  must be documented so nobody "cleans it up" mid-run.

## References

- `docs/contracts.md` § 1.3 — artifact layout and pointer format
- `docs/embedding-pipeline.md` § 7 — the full storage layout
- `docs/runbook.md` — Runbook 1, rollback procedure
- M2 milestone (`README.md`) — the eval gate that will read `current`
