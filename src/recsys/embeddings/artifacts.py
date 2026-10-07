"""Read and write embedding artifacts.

This module is the *only* place that reads or writes the embedding artifact
tree on disk. The contract is in ``docs/contracts.md`` § 1.3; the design and
rationale are in ``docs/embedding-pipeline.md`` § 7 and
``docs/adr/0004-versioned-runs-with-current-pointer.md``.

Layout::

    artifacts/embeddings/
    ├── runs/
    │   └── <run_id>/
    │       ├── embeddings.parquet
    │       ├── manifest.json
    │       └── state.json
    ├── current                    # one-line text file: the active run_id
    └── .lock                      # fcntl.flock target

Commit protocol (order is fixed, only step 4 makes a run visible):

    1. embeddings.parquet
    2. manifest.json
    3. state.json
    4. current

Concurrency: every mutation takes an exclusive ``fcntl.flock`` on ``.lock``.
POSIX only (Linux, macOS). NFS and most network filesystems are *not*
supported: ``flock`` semantics are unreliable there. See ADR-0004.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import pathlib
import socket
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import fcntl
except ImportError as _e:  # pragma: no cover - platform guard
    raise ImportError(
        "recsys.embeddings.artifacts requires POSIX fcntl (Linux, macOS). "
        "Windows is not supported; use WSL or Docker."
    ) from _e


# ============================================================================
# constants
# ============================================================================
SCHEMA_VERSION = 1
LOCK_TIMEOUT_DEFAULT_S = 30.0
NORM_TOLERANCE = 1e-4
BATCH_SIZE_DEFAULT = 50_000


# ============================================================================
# exceptions
# ============================================================================
class PipelineError(Exception):
    """Base class for all pipeline errors. Exit codes are mapped in pipeline.py."""


class OutputError(PipelineError):
    """Artifact writing, validation, or filesystem problem."""


class LockTimeoutError(PipelineError):
    """The run lock could not be acquired within the timeout."""


# ============================================================================
# paths
# ============================================================================
@dataclass(frozen=True)
class EmbeddingsRoot:
    """The root of the embeddings artifact tree, e.g. ``artifacts/embeddings``."""

    path: pathlib.Path

    @property
    def runs_dir(self) -> pathlib.Path:
        return self.path / "runs"

    @property
    def current_file(self) -> pathlib.Path:
        return self.path / "current"

    @property
    def lock_file(self) -> pathlib.Path:
        return self.path / ".lock"


@dataclass(frozen=True)
class RunPaths:
    """Pure path algebra for one run directory. No I/O."""

    root: EmbeddingsRoot
    run_id: str

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or "/" in self.run_id
            or "\\" in self.run_id
            or self.run_id.startswith(".")
        ):
            raise ValueError(f"unsafe run_id: {self.run_id!r}")

    @property
    def dir(self) -> pathlib.Path:
        return self.root.runs_dir / self.run_id

    @property
    def parquet(self) -> pathlib.Path:
        return self.dir / "embeddings.parquet"

    @property
    def manifest(self) -> pathlib.Path:
        return self.dir / "manifest.json"

    @property
    def state(self) -> pathlib.Path:
        return self.dir / "state.json"


@dataclass(frozen=True)
class CurrentRun:
    """The active run, as resolved from the ``current`` pointer."""

    run_id: str
    paths: RunPaths
    manifest: dict[str, Any]


# ============================================================================
# file lock
# ============================================================================
@contextmanager
def file_lock(
    path: pathlib.Path,
    *,
    timeout: float = LOCK_TIMEOUT_DEFAULT_S,
) -> Iterator[None]:
    """Exclusive advisory lock via :func:`fcntl.flock`.

    Blocks up to ``timeout`` seconds waiting for the lock. On success, writes
    a small diagnostic payload (pid, hostname, timestamp) into the file so a
    stuck holder can be identified from another process.

    ``flock`` is released automatically by the kernel when the holding
    process dies, so there is no stale-lock problem. NFS and most network
    filesystems do not honour ``flock`` reliably and are not supported —
    see ADR-0004.

    The lock file itself is left in place after a clean run; the lock is held
    by the file descriptor, not by the file's existence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+")
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno == errno.EINTR:
                    # Interrupted by a signal; retry immediately.
                    continue
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() >= deadline:
                    raise LockTimeoutError(
                        f"could not acquire lock on {path} within {timeout:.1f}s"
                    ) from None
                time.sleep(0.2)

        # We hold the lock. Write diagnostics; a failure here is not fatal.
        try:
            f.seek(0)
            f.truncate()
            f.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "acquired_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    sort_keys=True,
                )
            )
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            pass

        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


# ============================================================================
# config hash
# ============================================================================
def config_hash(
    *,
    preprocessing_version: str,
    onnx_artifact_sha256: str,
    model_name: str,
    max_seq_length: int,
    embedding_dim: int,
    pooling: str = "mean",
    normalize: bool = True,
) -> str:
    """Deterministic hash of parameters that change embedding *content*.

    Deliberately excludes thread count and batch size: those affect bit-level
    noise within the tolerance band defined in ``docs/embedding-pipeline.md``
    § 5, not the embedding's meaning. Including them would force a full
    re-encode on every CI run. Returns ``"sha256:<64 hex>"``.
    """
    payload: dict[str, Any] = {
        "preprocessing_version": preprocessing_version,
        "onnx_artifact_sha256": onnx_artifact_sha256,
        "model_name": model_name,
        "max_seq_length": int(max_seq_length),
        "embedding_dim": int(embedding_dim),
        "pooling": pooling,
        "normalize": bool(normalize),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ============================================================================
# run id allocation
# ============================================================================
def make_run_id(
    root: EmbeddingsRoot,
    *,
    when: datetime,
    model_version: str,
) -> str:
    """Return a unique run id of the form ``<ISO8601-Z>__<model_version>``.

    Two runs of the same model in the same second are legal — the strict
    determinism test does exactly that — so a ``-NN`` discriminator is
    appended on collision.
    """
    if when.tzinfo is None:
        raise ValueError("`when` must be timezone-aware")
    ts = when.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    base = f"{ts}__{model_version}"
    if not (root.runs_dir / base).exists():
        return base
    for i in range(1, 100):
        candidate = f"{ts}-{i:02d}__{model_version}"
        if not (root.runs_dir / candidate).exists():
            return candidate
    raise OutputError(f"could not allocate a unique run id for {base!r}")


# ============================================================================
# atomic file writes
# ============================================================================
def _fsync_dir(path: pathlib.Path) -> None:
    """Best-effort fsync of a directory, to make a rename durable.

    POSIX recommends fsyncing the containing directory after a rename so the
    rename survives a crash. Some filesystems do not support it; ignore those
    errors.
    """
    try:
        dfd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tempfile + fsync + rename)."""
    _atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: pathlib.Path, doc: dict[str, Any]) -> None:
    """Write ``doc`` as canonical JSON to ``path`` atomically."""
    _atomic_write_bytes(
        path,
        (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


# ============================================================================
# reading the current pointer and manifests
# ============================================================================
def read_current(root: EmbeddingsRoot) -> str | None:
    """Return the run_id named by ``current``, or ``None`` if absent/empty."""
    cf = root.current_file
    if not cf.is_file():
        return None
    content = cf.read_text(encoding="utf-8").strip()
    return content or None


def read_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Read and parse a manifest JSON file.

    Raises :class:`OutputError` if the file is missing, unreadable, not JSON,
    or not a JSON object.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise OutputError(f"could not read manifest {path}: {e}") from e
    if not isinstance(doc, dict):
        raise OutputError(f"manifest at {path} is not a JSON object")
    return doc


def resolve_current(root: EmbeddingsRoot) -> CurrentRun | None:
    """Follow ``current`` to the active run and return its manifest.

    Returns ``None`` if there is no active run. Raises :class:`OutputError`
    if the pointer names a run whose manifest is missing — a corrupted state
    that should never occur given the commit order (see ADR-0004).
    """
    run_id = read_current(root)
    if run_id is None:
        return None
    paths = RunPaths(root=root, run_id=run_id)
    if not paths.manifest.is_file():
        raise OutputError(f"current points at run {run_id!r} but {paths.manifest} is missing")
    return CurrentRun(run_id=run_id, paths=paths, manifest=read_manifest(paths.manifest))


# ============================================================================
# incremental state
# ============================================================================
@dataclass(frozen=True)
class StateRead:
    """Result of reading and validating an incremental state file."""

    state: dict[str, Any] | None
    reason: str | None  # populated only when ``state`` is None

    @property
    def valid(self) -> bool:
        return self.state is not None


def read_state(
    state_path: pathlib.Path,
    *,
    expected_model_version: str,
    expected_preprocessing_version: str,
    expected_config_hash: str,
) -> StateRead:
    """Load and validate an incremental state file.

    Returns :class:`StateRead`. If ``valid`` is False, ``reason`` names the
    first mismatch (``missing``, ``malformed``, ``schema_version``,
    ``model_version``, ``preprocessing_version``, ``config_hash``). Callers
    should treat an invalid state as "no previous state" and re-encode.

    See ``docs/embedding-pipeline.md`` § 7.1 for why each field must match:
    a model upgrade or preprocessing change must invalidate the state even
    when the catalog text is unchanged.
    """
    if not state_path.is_file():
        return StateRead(state=None, reason="missing")
    try:
        doc = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return StateRead(state=None, reason="malformed")
    if not isinstance(doc, dict):
        return StateRead(state=None, reason="malformed")
    if doc.get("schema_version") != SCHEMA_VERSION:
        return StateRead(state=None, reason="schema_version")
    if doc.get("model_version") != expected_model_version:
        return StateRead(state=None, reason="model_version")
    if doc.get("preprocessing_version") != expected_preprocessing_version:
        return StateRead(state=None, reason="preprocessing_version")
    if doc.get("config_hash") != expected_config_hash:
        return StateRead(state=None, reason="config_hash")
    items = doc.get("items")
    if not isinstance(items, dict):
        return StateRead(state=None, reason="malformed")
    return StateRead(state=doc, reason=None)


def read_state_items(state: dict[str, Any]) -> dict[str, str]:
    """Extract ``{item_id: content_hash}`` from a validated state document."""
    items = state.get("items")
    if not isinstance(items, dict):
        raise OutputError("state document has no items mapping")
    # Coerce to str for type safety; the writer always produces str->str.
    return {str(k): str(v) for k, v in items.items()}


# ============================================================================
# pre-commit validation
# ============================================================================
def validate_embeddings(
    *,
    embeddings: NDArray[np.float32],
    item_ids: list[str],
    expected_dim: int,
    atol: float = NORM_TOLERANCE,
) -> None:
    """Validate the assembled embedding table before it is written.

    Raises :class:`OutputError` on the first problem. This catches the class
    of bugs — off-by-one in the merge, wrong pooling, wrong model dim — that
    would otherwise reach retrieval and produce plausible but wrong results.
    See ``docs/embedding-pipeline.md`` § 7.4 for the checklist.
    """
    if embeddings.dtype != np.float32:
        raise OutputError(f"embeddings dtype must be float32, got {embeddings.dtype}")
    if embeddings.ndim != 2:
        raise OutputError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(item_ids):
        raise OutputError(
            f"row count mismatch: embeddings has {embeddings.shape[0]} rows "
            f"but {len(item_ids)} item_ids were provided"
        )
    if embeddings.shape[1] != expected_dim:
        raise OutputError(
            f"embedding dim mismatch: expected {expected_dim}, got {embeddings.shape[1]}"
        )
    if not np.isfinite(embeddings).all():
        raise OutputError("embeddings contain non-finite values (NaN or Inf)")
    if len(set(item_ids)) != len(item_ids):
        raise OutputError("item_ids are not unique")
    if embeddings.shape[0] == 0:
        return
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        bad = int(np.argmax(np.abs(norms - 1.0)))
        raise OutputError(
            f"embeddings are not L2-normalized: row {bad} has norm "
            f"{float(norms[bad]):.6f}, expected 1.0 ± {atol}"
        )


# ============================================================================
# small helpers
# ============================================================================
def content_hash_map_from_items(
    items: Sequence[Mapping[str, Any]],
    *,
    content_hash_fn: Any,
) -> dict[str, str]:
    """Return ``{item_id: content_hash}`` for catalog items.

    ``content_hash_fn`` is injected so this module does not import
    :mod:`recsys.embeddings.preprocess` at module load (it would drag in the
    encoder chain on every consumer of artifacts). ``items`` is typed as
    ``Sequence[Mapping[str, Any]]`` rather than ``list[dict[str, Any]]`` so
    TypedDict-shaped items (like ``CatalogItem``) are accepted without a cast.
    """
    return {str(it["item_id"]): str(content_hash_fn(it)) for it in items}


# ============================================================================
# manifest / state constructors
# ============================================================================
def build_manifest(
    *,
    run_id: str,
    created_at: str,
    mode: str,
    model_version: str,
    preprocessing_version: str,
    config_hash_value: str,
    catalog_snapshot: str,
    onnx_artifact_sha256: str,
    parquet_relative_path: str,
    parquet_sha256: str,
    parquet_rows: int,
    parquet_encoded_rows: int,
    parquet_dim: int,
    parquet_dtype: str,
    state_relative_path: str,
    state_sha256: str,
    environment: dict[str, str | None],
) -> dict[str, Any]:
    """Build the manifest document for a run.

    Paths must be relative to the embeddings root so the tree can be moved
    without rewriting the manifest.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "run_id": run_id,
        "mode": mode,
        "model_version": model_version,
        "preprocessing_version": preprocessing_version,
        "config_hash": config_hash_value,
        "catalog_snapshot": catalog_snapshot,
        "onnx_artifact_sha256": onnx_artifact_sha256,
        "parquet": {
            "path": parquet_relative_path,
            "sha256": parquet_sha256,
            "rows": int(parquet_rows),
            "encoded_rows": int(parquet_encoded_rows),
            "dim": int(parquet_dim),
            "dtype": parquet_dtype,
        },
        "state": {
            "path": state_relative_path,
            "sha256": state_sha256,
        },
        "environment": dict(environment),
    }


def build_state(
    *,
    model_version: str,
    preprocessing_version: str,
    config_hash_value: str,
    catalog_snapshot: str,
    items: dict[str, str],
) -> dict[str, Any]:
    """Build the incremental state document for a run."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "preprocessing_version": preprocessing_version,
        "config_hash": config_hash_value,
        "hash_algorithm": "sha256",
        "catalog_snapshot": catalog_snapshot,
        "items": dict(items),
    }


# ============================================================================
# commit
# ============================================================================
def commit_run(
    *,
    root: EmbeddingsRoot,
    run_paths: RunPaths,
    manifest: dict[str, Any],
    state_doc: dict[str, Any],
) -> None:
    """Commit a run whose Parquet has already been written.

    Writes ``manifest.json`` and ``state.json`` atomically inside the run
    directory, then atomically updates the ``current`` pointer. The Parquet
    must already exist at :attr:`RunPaths.parquet` (written by the caller).

    Order is fixed (see ADR-0004 and ``docs/embedding-pipeline.md`` § 7.2):

    1. Parquet      (caller)
    2. manifest     (here)
    3. state        (here)
    4. current      (here)

    Only step 4 makes the run visible to consumers. A crash before it leaves
    the previous ``current`` in place; the orphan run directory is ignored.
    """
    if not run_paths.parquet.is_file():
        raise OutputError(f"cannot commit: parquet not found at {run_paths.parquet}")
    atomic_write_json(run_paths.manifest, manifest)
    atomic_write_json(run_paths.state, state_doc)
    atomic_write_text(root.current_file, run_paths.run_id + "\n")


# ============================================================================
# pyarrow guards (lazy: this module imports without the 'pipeline' extra)
# ============================================================================
def _pa() -> Any:
    from recsys._optional import optional_import

    mod = optional_import("pyarrow")
    if mod is None:
        raise RuntimeError(
            "Embedding artifact I/O requires the 'pipeline' extra. "
            "Install with: pip install -e '.[pipeline]'"
        )
    return mod


def _pq() -> Any:
    from recsys._optional import optional_import

    mod = optional_import("pyarrow.parquet")
    if mod is None:
        raise RuntimeError(
            "Embedding artifact I/O requires the 'pipeline' extra. "
            "Install with: pip install -e '.[pipeline]'"
        )
    return mod


# ============================================================================
# streaming read of a previous run's Parquet
# ============================================================================
@dataclass(frozen=True)
class PreviousBatch:
    """One batch of rows read from a previous run's Parquet."""

    item_ids: list[str]
    embeddings: NDArray[np.float32]
    content_hashes: list[str]


def iter_previous_batches(
    parquet: pathlib.Path,
    *,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> Iterator[PreviousBatch]:
    """Yield the previous run's rows as :class:`PreviousBatch` objects.

    Reads one row group batch at a time via
    :meth:`pyarrow.parquet.ParquetFile.iter_batches`. Peak memory is
    ``batch_size * dim * 4`` bytes for the embedding slice, plus the string
    columns. See ``docs/embedding-pipeline.md`` § 7.5.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pq = _pq()
    pf = pq.ParquetFile(str(parquet))
    # A Parquet with zero row groups (e.g. written by a test that only
    # exercised an empty batch) has no rows. pyarrow 17 returned an empty
    # iterator; pyarrow 18 raises "requested metadata for row group: -1"
    # when the reader is asked for a batch. Checking up front is
    # version-independent and states the intent.
    if pf.metadata is None or pf.metadata.num_row_groups == 0:
        return
    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=["item_id", "embedding", "content_hash"],
    ):
        n = batch.num_rows
        if n == 0:
            continue
        ids = batch.column("item_id").to_pylist()
        hashes = batch.column("content_hash").to_pylist()
        # `embedding` is a ListArray<float32>; flatten to a 1-D numpy array
        # and reshape, avoiding a per-row Python list intermediate.
        flat = batch.column("embedding").flatten()
        arr = flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        if arr.size % n != 0:
            raise OutputError(f"parquet batch is ragged: {arr.size} values across {n} rows")
        dim = arr.size // n
        emb = arr.reshape(n, dim)
        yield PreviousBatch(
            item_ids=list(ids),
            embeddings=emb,
            content_hashes=list(hashes),
        )


# ============================================================================
# streaming Parquet writer
# ============================================================================
class ParquetBatchWriter:
    """Incremental Parquet writer with a fixed schema and zstd compression.

    One row group per :meth:`write_batch`. Use as a context manager; the
    file is finalised on exit. The schema matches the contract in
    ``docs/contracts.md`` § 1.3:

        item_id               string
        embedding             list<float32>[dim]
        content_hash          string
        preprocessing_version string
    """

    def __init__(self, path: pathlib.Path, *, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._path = path
        self._dim = embedding_dim
        self._writer: Any = None
        self._schema: Any = None

    def __enter__(self) -> ParquetBatchWriter:
        pa = _pa()
        pq = _pq()
        self._schema = pa.schema(
            [
                ("item_id", pa.string()),
                ("embedding", pa.list_(pa.float32(), list_size=self._dim)),
                ("content_hash", pa.string()),
                ("preprocessing_version", pa.string()),
            ]
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(
            str(self._path),
            self._schema,
            compression="zstd",
            version="2.6",
            use_dictionary=["item_id", "content_hash", "preprocessing_version"],
            write_statistics=False,
        )
        return self

    def write_batch(
        self,
        *,
        item_ids: list[str],
        embeddings: NDArray[np.float32],
        content_hashes: list[str],
        preprocessing_version: str,
    ) -> None:
        """Append one row group. Silently ignores empty batches."""
        if self._writer is None or self._schema is None:
            raise RuntimeError("ParquetBatchWriter must be used as a context manager")
        if not item_ids:
            return
        if len(item_ids) != len(content_hashes):
            raise OutputError(
                f"batch length mismatch: item_ids={len(item_ids)} "
                f"content_hashes={len(content_hashes)}"
            )
        if embeddings.shape[0] != len(item_ids):
            raise OutputError(
                f"batch length mismatch: item_ids={len(item_ids)} "
                f"embeddings rows={embeddings.shape[0]}"
            )
        if embeddings.shape[1] != self._dim:
            raise OutputError(f"batch embedding dim {embeddings.shape[1]} != expected {self._dim}")
        pa = _pa()
        rows = [
            {
                "item_id": item_ids[i],
                "embedding": embeddings[i].tolist(),
                "content_hash": content_hashes[i],
                "preprocessing_version": preprocessing_version,
            }
            for i in range(len(item_ids))
        ]
        table = pa.Table.from_pylist(rows, schema=self._schema)
        self._writer.write_table(table)

    def __exit__(self, *exc: object) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


__all__ = [
    "BATCH_SIZE_DEFAULT",
    "LOCK_TIMEOUT_DEFAULT_S",
    "NORM_TOLERANCE",
    "SCHEMA_VERSION",
    "CurrentRun",
    "EmbeddingsRoot",
    "LockTimeoutError",
    "OutputError",
    "ParquetBatchWriter",
    "PipelineError",
    "PreviousBatch",
    "RunPaths",
    "StateRead",
    "atomic_write_json",
    "atomic_write_text",
    "build_manifest",
    "build_state",
    "commit_run",
    "config_hash",
    "content_hash_map_from_items",
    "file_lock",
    "iter_previous_batches",
    "make_run_id",
    "read_current",
    "read_manifest",
    "read_state",
    "read_state_items",
    "resolve_current",
    "validate_embeddings",
]
