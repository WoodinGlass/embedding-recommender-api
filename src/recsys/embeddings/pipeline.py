"""Batch and incremental embedding pipeline.

The CLI is the integration surface for CI and M6 orchestration. Stdout is a
single JSON line (the summary); progress per batch goes to stderr as JSON
lines and can be silenced with ``--quiet``. See ``docs/embedding-pipeline.md``
§ 6.

Two modes, one primitive:

- **batch**: encode every item. Always writes a new run.
- **incremental**: encode only items whose ``content_hash`` changed since the
  previous run; reuse the rest by streaming the previous Parquet. If nothing
  changed — same catalog snapshot, same config hash, same content hashes —
  the pipeline exits 0 with ``status="no_changes"`` and leaves the active run
  untouched.

Concurrency and commit order are enforced by :mod:`recsys.embeddings.artifacts`.
The lock is taken before reading state or writing anything, and released when
the process ends. See ADR-0004.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from recsys.embeddings.artifacts import (
    LOCK_TIMEOUT_DEFAULT_S,
    EmbeddingsRoot,
    LockTimeoutError,
    OutputError,
    ParquetBatchWriter,
    PipelineError,
    RunPaths,
    StateRead,
    build_manifest,
    build_state,
    commit_run,
    content_hash_map_from_items,
    file_lock,
    iter_previous_batches,
    make_run_id,
    read_state,
    resolve_current,
    validate_embeddings,
)
from recsys.embeddings.artifacts import (
    config_hash as compute_config_hash,
)
from recsys.embeddings.encoder import Encoder, OnnxEncoder, sha256_file
from recsys.embeddings.preprocess import (
    PREPROCESSING_VERSION,
    CatalogItem,
    catalog_snapshot,
    item_content_hash,
    preprocess_item,
)

# --------------------------------------------------------------------------- #
# exit codes (documented in docs/embedding-pipeline.md § 6)
# --------------------------------------------------------------------------- #
EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_ENCODER_ERROR = 3
EXIT_OUTPUT_ERROR = 4
EXIT_LOCK_TIMEOUT = 5


class InputError(Exception):
    """Raised for catalog or argument problems. Maps to exit code 2."""


# --------------------------------------------------------------------------- #
# catalog loading
# --------------------------------------------------------------------------- #
_REQUIRED_FIELDS = frozenset({"item_id", "title", "description", "category", "brand"})


def load_catalog(path: pathlib.Path) -> list[CatalogItem]:
    """Read a JSONL catalog. Raises :class:`InputError` on malformed input.

    Duplicate ``item_id`` is a hard error: an ambiguous catalog is a bug in
    the source, not something the pipeline should paper over.
    """
    if not path.is_file():
        raise InputError(f"catalog not found: {path}")
    items: list[CatalogItem] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise InputError(f"line {lineno}: not a JSON object")
        missing = _REQUIRED_FIELDS - set(obj.keys())
        if missing:
            raise InputError(f"line {lineno}: missing fields {sorted(missing)}")
        iid = obj["item_id"]
        if not isinstance(iid, str) or not iid:
            raise InputError(f"line {lineno}: item_id must be a non-empty string")
        if iid in seen:
            raise InputError(f"line {lineno}: duplicate item_id {iid!r}")
        seen.add(iid)
        items.append(
            CatalogItem(
                item_id=iid,
                title=str(obj["title"]),
                description=str(obj["description"]),
                category=str(obj["category"]),
                brand=str(obj["brand"]),
            )
        )
    if not items:
        raise InputError(f"catalog {path} contains no items")
    return items


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Plan:
    """Which items to encode and which are reused from the previous run."""

    sorted_items: list[CatalogItem]
    items_to_encode: list[CatalogItem]
    unchanged_ids: frozenset[str]
    content_hashes: dict[str, str]
    catalog_snapshot: str
    mode: str
    state_reason: str | None  # non-None when the state was invalidated
    no_changes: bool


def plan_batch(items: list[CatalogItem]) -> Plan:
    """Batch: encode everything, always write a new run."""
    sorted_items = sorted(items, key=lambda it: it["item_id"])
    hashes = content_hash_map_from_items(sorted_items, content_hash_fn=item_content_hash)
    return Plan(
        sorted_items=sorted_items,
        items_to_encode=list(sorted_items),
        unchanged_ids=frozenset(),
        content_hashes=hashes,
        catalog_snapshot=catalog_snapshot(sorted_items),
        mode="batch",
        state_reason=None,
        no_changes=False,
    )


def plan_incremental(
    items: list[CatalogItem],
    *,
    state_read: StateRead,
) -> Plan:
    """Incremental: reuse unchanged embeddings, encode the rest.

    A state is a *candidate* for reuse only if it passed validation in
    :func:`recsys.embeddings.artifacts.read_state`. This function additionally
    requires ``state["catalog_snapshot"]`` to equal the new catalog snapshot
    for the run to be considered a no-op.
    """
    sorted_items = sorted(items, key=lambda it: it["item_id"])
    hashes = content_hash_map_from_items(sorted_items, content_hash_fn=item_content_hash)
    snap = catalog_snapshot(sorted_items)

    if not state_read.valid:
        # No usable previous state: encode everything.
        return Plan(
            sorted_items=sorted_items,
            items_to_encode=list(sorted_items),
            unchanged_ids=frozenset(),
            content_hashes=hashes,
            catalog_snapshot=snap,
            mode="incremental",
            state_reason=state_read.reason or "invalid",
            no_changes=False,
        )

    assert state_read.state is not None  # for mypy; guarded by `valid`
    prev_items = state_read.state.get("items")
    if not isinstance(prev_items, dict):
        # read_state already validated this, but keep the guard for safety.
        return Plan(
            sorted_items=sorted_items,
            items_to_encode=list(sorted_items),
            unchanged_ids=frozenset(),
            content_hashes=hashes,
            catalog_snapshot=snap,
            mode="incremental",
            state_reason="malformed",
            no_changes=False,
        )

    unchanged: set[str] = set()
    to_encode: list[CatalogItem] = []
    for it in sorted_items:
        iid = it["item_id"]
        prev_hash = prev_items.get(iid)
        if isinstance(prev_hash, str) and prev_hash == hashes[iid]:
            unchanged.add(iid)
        else:
            to_encode.append(it)

    # The run is a no-op only if nothing changed *and* the state was written
    # for this exact catalog snapshot. If snapshot differs but all hashes
    # match, something structural changed (an item was added and another
    # removed with the same net set of hashes — impossible with unique IDs,
    # but the check is cheap and makes the invariant explicit).
    prev_snap = state_read.state.get("catalog_snapshot")
    no_changes = not to_encode and prev_snap == snap

    return Plan(
        sorted_items=sorted_items,
        items_to_encode=to_encode,
        unchanged_ids=frozenset(unchanged),
        content_hashes=hashes,
        catalog_snapshot=snap,
        mode="incremental",
        state_reason=None,
        no_changes=no_changes,
    )


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def encode_lazily(
    encoder: Encoder,
    items: list[CatalogItem],
    *,
    batch_size: int,
    progress: bool,
) -> Iterator[NDArray[np.float32]]:
    """Yield embeddings one row at a time, encoding in batches.

    Peak memory is ``batch_size * dim * 4`` bytes for the current batch. The
    caller is expected to consume rows in order; ``items`` must already be
    sorted by ``item_id`` so the yielded rows align with the output stream.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total = len(items)
    for start in range(0, total, batch_size):
        chunk = items[start : start + batch_size]
        texts = [preprocess_item(it) for it in chunk]
        emb = encoder.encode(texts)
        if emb.shape != (len(chunk), encoder.embedding_dim):
            raise OutputError(
                f"encoder returned shape {emb.shape}, expected "
                f"({len(chunk)}, {encoder.embedding_dim})"
            )
        if progress:
            _log(
                {
                    "event": "pipeline.batch",
                    "batch_index": start // batch_size,
                    "total_batches": (total + batch_size - 1) // batch_size,
                    "batch_size": len(chunk),
                    "encoded_total": min(start + len(chunk), total),
                    "of": total,
                }
            )
        for row in range(emb.shape[0]):
            yield emb[row]


# --------------------------------------------------------------------------- #
# streaming merge
# --------------------------------------------------------------------------- #
def _stream_merge(
    *,
    plan: Plan,
    encoder: Encoder,
    previous_parquet: pathlib.Path | None,
    output_parquet: pathlib.Path,
    batch_size: int,
    progress: bool,
) -> None:
    """Write the output Parquet by merging reused and freshly-encoded rows.

    Walks three ordered streams in lockstep:

    - ``plan.sorted_items`` — every item, in ``item_id`` order.
    - the previous Parquet (if any) — items in ``item_id`` order, with rows
      for items no longer in the catalog skipped.
    - the encoder — yields embeddings for ``plan.items_to_encode`` in
      ``item_id`` order.

    Neither the previous embeddings nor the new embeddings are ever fully
    materialised. Peak memory is O(batch_size * dim).
    """
    dim = encoder.embedding_dim
    encode_iter = encode_lazily(
        encoder,
        plan.items_to_encode,
        batch_size=batch_size,
        progress=progress,
    )

    prev_iter = (
        iter_previous_batches(previous_parquet, batch_size=batch_size)
        if previous_parquet is not None
        else iter(())
    )
    prev_batch = next(prev_iter, None)
    prev_idx = 0

    def take_previous(iid: str) -> NDArray[np.float32]:
        """Return the previous embedding for ``iid``, advancing the iterator.

        Items present in the previous Parquet but absent from the current
        catalog (deletions) are skipped. A missing item at this point means
        the state claimed the item was unchanged but its embedding is gone,
        which is a corruption and must fail loudly.
        """
        nonlocal prev_batch, prev_idx
        while True:
            if prev_batch is None or prev_idx >= len(prev_batch.item_ids):
                prev_batch = next(prev_iter, None)
                prev_idx = 0
                if prev_batch is None:
                    raise OutputError(
                        f"item {iid!r} not found in previous Parquet; state "
                        f"is inconsistent with the active run"
                    )
            got = prev_batch.item_ids[prev_idx]
            if got == iid:
                # `[i, :]` (not `[i]`) so mypy sees a 1-D array slice rather
                # than a scalar indexing result.
                emb: NDArray[np.float32] = prev_batch.embeddings[prev_idx, :]
                prev_idx += 1
                return emb
            if got < iid:
                prev_idx += 1  # deleted item — skip
                continue
            raise OutputError(
                f"previous Parquet yielded {got!r} but {iid!r} was expected; "
                f"the previous run is not sorted by item_id or is corrupt"
            )

    buffer_ids: list[str] = []
    buffer_embs: list[NDArray[np.float32]] = []
    buffer_hashes: list[str] = []

    def flush(writer: ParquetBatchWriter) -> None:
        if not buffer_ids:
            return
        arr = np.stack(buffer_embs, axis=0).astype(np.float32, copy=False)
        # Pre-commit validation, per batch. See docs/embedding-pipeline.md
        # § 7.4. Running it here — rather than on a fully materialised table —
        # keeps peak memory at one batch while still catching the class of
        # bugs (NaN/Inf, wrong dim, denormalized rows, duplicate ids) that
        # would otherwise reach retrieval unnoticed.
        validate_embeddings(
            embeddings=arr,
            item_ids=list(buffer_ids),
            expected_dim=dim,
        )
        writer.write_batch(
            item_ids=list(buffer_ids),
            embeddings=arr,
            content_hashes=list(buffer_hashes),
            preprocessing_version=PREPROCESSING_VERSION,
        )
        buffer_ids.clear()
        buffer_embs.clear()
        buffer_hashes.clear()

    with ParquetBatchWriter(output_parquet, embedding_dim=dim) as writer:
        for it in plan.sorted_items:
            iid = it["item_id"]
            emb = take_previous(iid) if iid in plan.unchanged_ids else next(encode_iter)
            buffer_ids.append(iid)
            buffer_embs.append(emb)
            buffer_hashes.append(plan.content_hashes[iid])
            if len(buffer_ids) >= batch_size:
                flush(writer)
        flush(writer)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunSummary:
    """Returned by :func:`run`; serialised as one JSON line by the CLI."""

    status: str  # "ok" | "no_changes"
    mode: str
    run_id: str
    model_version: str
    config_hash: str
    catalog_snapshot: str
    total: int
    unchanged: int
    encoded: int
    deleted: int
    duration_ms: int
    output_path: str
    checksum: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "event": "pipeline.done",
            "status": self.status,
            "mode": self.mode,
            "run_id": self.run_id,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "catalog_snapshot": self.catalog_snapshot,
            "total": self.total,
            "unchanged": self.unchanged,
            "changed": self.encoded,
            "new": 0,  # filled by caller when distinguishable; 0 for now
            "deleted": self.deleted,
            "encoded": self.encoded,
            "duration_ms": self.duration_ms,
            "output_path": self.output_path,
            "checksum": self.checksum,
        }


def _env_versions() -> dict[str, str | None]:
    """Best-effort environment versions recorded in the manifest."""
    import platform

    def _v(modname: str) -> str | None:
        try:
            mod = __import__(modname)
        except ImportError:
            return None
        return getattr(mod, "__version__", None)

    return {
        "python_version": platform.python_version(),
        "onnxruntime_version": _v("onnxruntime"),
        "numpy_version": _v("numpy"),
        "pyarrow_version": _v("pyarrow"),
    }


def _config_hash_for(encoder: Encoder, onnx_artifact_sha256: str) -> str:
    """Compute the config_hash from the encoder's content-affecting knobs."""
    # These attributes exist on OnnxEncoder; the ReferenceEncoder used by
    # tests has no ONNX artifact, so callers pass a stable substitute.
    model_name = getattr(encoder, "model_name", encoder.model_version)
    max_seq_length = int(getattr(encoder, "max_seq_length", 512))
    return compute_config_hash(
        preprocessing_version=PREPROCESSING_VERSION,
        onnx_artifact_sha256=onnx_artifact_sha256,
        model_name=str(model_name),
        max_seq_length=max_seq_length,
        embedding_dim=encoder.embedding_dim,
    )


def run(
    *,
    catalog_path: pathlib.Path,
    out_dir: pathlib.Path,
    encoder: Encoder,
    mode: str,
    batch_size: int,
    onnx_artifact_sha256: str,
    lock_timeout: float = LOCK_TIMEOUT_DEFAULT_S,
    progress: bool = True,
) -> RunSummary:
    """Run one pipeline invocation and commit the result.

    Caller supplies the encoder and the ONNX artifact SHA256 so this function
    is testable without touching ``artifacts/onnx`` and without importing
    onnxruntime.

    Concurrency: an exclusive ``flock`` is held for the entire run, from
    before the state is read to after the pointer is updated.
    """
    if mode not in ("incremental", "batch"):
        raise InputError(f"unknown mode: {mode!r}")

    t0 = time.perf_counter()
    root = EmbeddingsRoot(path=out_dir)

    with file_lock(root.lock_file, timeout=lock_timeout):
        items = load_catalog(catalog_path)
        cfg_hash = _config_hash_for(encoder, onnx_artifact_sha256)
        current = resolve_current(root)

        # Item ids recorded in the currently active state (if any).
        # Used to compute the "deleted" count once the new plan is
        # known. Empty when there is no valid state.
        previous_item_ids: set[str] = set()

        if mode == "batch":
            plan = plan_batch(items)
        else:
            previous_state: StateRead
            if current is None:
                previous_state = StateRead(state=None, reason="missing")
            else:
                previous_state = read_state(
                    current.paths.state,
                    expected_model_version=encoder.model_version,
                    expected_preprocessing_version=PREPROCESSING_VERSION,
                    expected_config_hash=cfg_hash,
                )
                if not previous_state.valid:
                    _log(
                        {
                            "event": "pipeline.state.invalidated",
                            "reason": previous_state.reason,
                        }
                    )
                else:
                    assert previous_state.state is not None  # guarded by valid
                    items_map = previous_state.state.get("items")
                    if isinstance(items_map, dict):
                        previous_item_ids = set(items_map.keys())
            plan = plan_incremental(items, state_read=previous_state)

        # ----- no-op path ---------------------------------------------------
        if plan.no_changes and current is not None:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            parquet_rel = current.manifest.get("parquet", {})
            parquet_path = parquet_rel.get("path") if isinstance(parquet_rel, dict) else None
            checksum = parquet_rel.get("sha256") if isinstance(parquet_rel, dict) else None
            summary = RunSummary(
                status="no_changes",
                mode=plan.mode,
                run_id=current.run_id,
                model_version=encoder.model_version,
                config_hash=cfg_hash,
                catalog_snapshot=plan.catalog_snapshot,
                total=len(plan.sorted_items),
                unchanged=len(plan.unchanged_ids),
                encoded=0,
                deleted=0,
                duration_ms=duration_ms,
                output_path=str(root.path / str(parquet_path)) if parquet_path else "",
                checksum=str(checksum) if checksum else None,
            )
            if progress:
                _log({"event": "pipeline.no_changes", "run_id": current.run_id})
            return summary

        # ----- write path ---------------------------------------------------
        when = datetime.now(UTC)
        run_id = make_run_id(root, when=when, model_version=encoder.model_version)
        run_paths = RunPaths(root=root, run_id=run_id)

        previous_parquet: pathlib.Path | None = None
        if plan.mode == "incremental" and plan.unchanged_ids:
            if current is None:
                raise OutputError("incremental run has unchanged items but no active run")
            previous_parquet = current.paths.parquet

        _stream_merge(
            plan=plan,
            encoder=encoder,
            previous_parquet=previous_parquet,
            output_parquet=run_paths.parquet,
            batch_size=batch_size,
            progress=progress,
        )

        parquet_sha = sha256_file(run_paths.parquet)

        state_doc = build_state(
            model_version=encoder.model_version,
            preprocessing_version=PREPROCESSING_VERSION,
            config_hash_value=cfg_hash,
            catalog_snapshot=plan.catalog_snapshot,
            items=plan.content_hashes,
        )
        state_bytes = (json.dumps(state_doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
        import hashlib as _hashlib

        state_sha = _hashlib.sha256(state_bytes).hexdigest()

        manifest = build_manifest(
            run_id=run_id,
            created_at=when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            mode=plan.mode,
            model_version=encoder.model_version,
            preprocessing_version=PREPROCESSING_VERSION,
            config_hash_value=cfg_hash,
            catalog_snapshot=plan.catalog_snapshot,
            onnx_artifact_sha256=onnx_artifact_sha256,
            parquet_relative_path=f"runs/{run_id}/embeddings.parquet",
            parquet_sha256=parquet_sha,
            parquet_rows=len(plan.sorted_items),
            parquet_encoded_rows=len(plan.items_to_encode),
            parquet_dim=encoder.embedding_dim,
            parquet_dtype="float32",
            state_relative_path=f"runs/{run_id}/state.json",
            state_sha256=state_sha,
            environment=_env_versions(),
        )

        commit_run(
            root=root,
            run_paths=run_paths,
            manifest=manifest,
            state_doc=state_doc,
        )

        duration_ms = int((time.perf_counter() - t0) * 1000)
        return RunSummary(
            status="ok",
            mode=plan.mode,
            run_id=run_id,
            model_version=encoder.model_version,
            config_hash=cfg_hash,
            catalog_snapshot=plan.catalog_snapshot,
            total=len(plan.sorted_items),
            unchanged=len(plan.unchanged_ids),
            encoded=len(plan.items_to_encode),
            deleted=len(previous_item_ids - {it["item_id"] for it in plan.sorted_items}),
            duration_ms=duration_ms,
            output_path=str(run_paths.parquet),
            checksum=parquet_sha,
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _log(doc: dict[str, Any]) -> None:
    """Emit one JSON line to stderr."""
    print(json.dumps(doc), file=sys.stderr, flush=True)


def _build_encoder(args: argparse.Namespace) -> Encoder:
    """Construct the ONNX encoder from settings or CLI overrides."""
    from recsys.config.settings import get_settings

    settings = get_settings()
    onnx_dir = (
        pathlib.Path(args.onnx_dir) if args.onnx_dir else pathlib.Path(settings.embedding_onnx_path)
    )
    return OnnxEncoder(onnx_dir, label=args.label)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-embed",
        description=(
            "Batch and incremental embedding pipeline. Writes a versioned "
            "run under <out>/runs/<run_id>/ and updates <out>/current."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("incremental", "batch"),
        default="incremental",
        help="incremental (default) or batch",
    )
    parser.add_argument(
        "--catalog",
        type=pathlib.Path,
        default=pathlib.Path("data/sample/catalog.jsonl"),
        help="JSONL catalog file",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/embeddings"),
        help="root of the embeddings artifact tree (contains runs/, current, .lock)",
    )
    parser.add_argument(
        "--onnx-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "directory containing model.onnx and its sidecars "
            "(default: EMBEDDING_ONNX_PATH from settings)"
        ),
    )
    parser.add_argument(
        "--label",
        default="minilm-onnx-v1",
        help="human-readable model label used in model_version",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="encode batch size (default: EMBEDDING_BATCH_SIZE from settings)",
    )
    parser.add_argument(
        "--lock-timeout",
        type=float,
        default=LOCK_TIMEOUT_DEFAULT_S,
        help="seconds to wait for the run lock before exiting with code 5",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-batch progress lines on stderr",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    from recsys.config.settings import get_settings

    settings = get_settings()
    batch_size = args.batch_size or settings.embedding_batch_size

    try:
        encoder = _build_encoder(args)
    except Exception as exc:
        # Defensive boundary: any encoder-construction failure (missing
        # artifact, onnxruntime mismatch, bad label) becomes a single
        # structured error line and a distinct exit code.
        _log({"event": "pipeline.encoder.error", "message": str(exc)})
        return EXIT_ENCODER_ERROR

    onnx_sha = getattr(encoder, "sha256", "0" * 64)

    try:
        summary = run(
            catalog_path=args.catalog,
            out_dir=args.out,
            encoder=encoder,
            mode=args.mode,
            batch_size=batch_size,
            onnx_artifact_sha256=onnx_sha,
            lock_timeout=args.lock_timeout,
            progress=not args.quiet,
        )
    except InputError as exc:
        _log({"event": "pipeline.input.error", "message": str(exc)})
        return EXIT_INPUT_ERROR
    except LockTimeoutError as exc:
        _log({"event": "pipeline.locked", "message": str(exc)})
        return EXIT_LOCK_TIMEOUT
    except (OutputError, PipelineError) as exc:
        _log({"event": "pipeline.output.error", "message": str(exc)})
        return EXIT_OUTPUT_ERROR

    print(json.dumps(summary.to_json()))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
