"""End-to-end pipeline tests — integration, encoder tier.

Exercises the M1 exit criteria against the real ONNX encoder:

1. A batch run writes a complete run directory and updates ``current``.
2. Two batch runs produce **byte-identical** Parquet (strict determinism).
3. An incremental run over an unchanged catalog matches batch (semantic
   determinism): the incremental path is a performance optimisation, not
   a different result.
4. A second incremental run over the same catalog reports ``no_changes``
   and does not touch the active run.
5. A single edited item causes exactly that item to be re-encoded.
6. The CLI writes exactly one JSON line to stdout (progress goes to stderr).
7. A held lock causes the next run to time out with ``LockTimeoutError``.

Requires the ``[inference,export,pipeline]`` extras. The ONNX artifact is
exported on first run and reused thereafter. Marked ``encoder`` so CI can
run it in a dedicated job (see ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import json
import pathlib
import time

import pyarrow.parquet as pq
import pytest

from recsys.embeddings import artifacts as art
from recsys.embeddings import pipeline as pl
from recsys.embeddings.encoder import OnnxEncoder, sha256_file
from recsys.embeddings.onnx_export import export

pytestmark = [pytest.mark.integration, pytest.mark.encoder]

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_SLUG = "sentence-transformers__all-MiniLM-L6-v2"
LABEL = "minilm-onnx-v1"
ONNX_DIR = REPO_ROOT / "artifacts" / "onnx" / MODEL_SLUG
CATALOG = REPO_ROOT / "data" / "sample" / "catalog.jsonl"
EXPECTED_ROWS = 200


@pytest.fixture(scope="module")
def onnx_dir() -> pathlib.Path:
    if not (ONNX_DIR / "model.onnx").is_file():
        export(MODEL_NAME, ONNX_DIR)
    return ONNX_DIR


@pytest.fixture(scope="module")
def encoder(onnx_dir: pathlib.Path) -> OnnxEncoder:
    return OnnxEncoder(onnx_dir, label=LABEL)


@pytest.fixture(scope="module")
def onnx_sha(encoder: OnnxEncoder) -> str:
    return encoder.sha256


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run(
    *,
    catalog: pathlib.Path,
    out_dir: pathlib.Path,
    mode: str,
    encoder: OnnxEncoder,
    onnx_sha: str,
    batch_size: int = 64,
    lock_timeout: float = 5.0,
) -> pl.RunSummary:
    return pl.run(
        catalog_path=catalog,
        out_dir=out_dir,
        encoder=encoder,
        mode=mode,
        batch_size=batch_size,
        onnx_artifact_sha256=onnx_sha,
        lock_timeout=lock_timeout,
        progress=False,
    )


def _read_current(root: art.EmbeddingsRoot) -> str:
    run_id = art.read_current(root)
    assert run_id is not None, "current pointer is empty"
    return run_id


# --------------------------------------------------------------------------- #
# 1. structure of a batch run
# --------------------------------------------------------------------------- #
def test_batch_run_writes_full_run_directory(
    tmp_path: pathlib.Path, encoder: OnnxEncoder, onnx_sha: str
) -> None:
    out = tmp_path / "emb"
    summary = _run(
        catalog=CATALOG,
        out_dir=out,
        mode="batch",
        encoder=encoder,
        onnx_sha=onnx_sha,
    )
    assert summary.status == "ok"
    assert summary.mode == "batch"
    assert summary.total == EXPECTED_ROWS
    assert summary.encoded == EXPECTED_ROWS
    assert summary.unchanged == 0

    root = art.EmbeddingsRoot(path=out)
    run_paths = art.RunPaths(root=root, run_id=summary.run_id)

    # Run directory contents
    assert run_paths.parquet.is_file()
    assert run_paths.manifest.is_file()
    assert run_paths.state.is_file()

    # current pointer names this run
    assert _read_current(root) == summary.run_id

    # Parquet schema and rows match the contract
    table = pq.read_table(str(run_paths.parquet))
    assert table.column_names == [
        "item_id",
        "embedding",
        "content_hash",
        "preprocessing_version",
    ]
    assert table.num_rows == EXPECTED_ROWS

    # Manifest is self-consistent
    manifest = art.read_manifest(run_paths.manifest)
    assert manifest["schema_version"] == art.SCHEMA_VERSION
    assert manifest["run_id"] == summary.run_id
    assert manifest["parquet"]["rows"] == EXPECTED_ROWS
    assert manifest["parquet"]["encoded_rows"] == EXPECTED_ROWS
    assert manifest["parquet"]["dim"] == encoder.embedding_dim
    assert manifest["parquet"]["sha256"] == sha256_file(run_paths.parquet)
    # Paths are relative, never absolute
    assert not manifest["parquet"]["path"].startswith("/")
    assert not manifest["state"]["path"].startswith("/")


# --------------------------------------------------------------------------- #
# 2. strict determinism: two batch runs → byte-identical Parquet
# --------------------------------------------------------------------------- #
def test_two_batch_runs_produce_byte_identical_parquet(
    tmp_path: pathlib.Path, encoder: OnnxEncoder, onnx_sha: str
) -> None:
    out1 = tmp_path / "emb1"
    out2 = tmp_path / "emb2"

    s1 = _run(catalog=CATALOG, out_dir=out1, mode="batch", encoder=encoder, onnx_sha=onnx_sha)
    s2 = _run(catalog=CATALOG, out_dir=out2, mode="batch", encoder=encoder, onnx_sha=onnx_sha)

    assert s1.checksum is not None
    assert s2.checksum is not None
    # Each run's own SHA matches its own parquet
    p1 = art.RunPaths(root=art.EmbeddingsRoot(path=out1), run_id=s1.run_id).parquet
    p2 = art.RunPaths(root=art.EmbeddingsRoot(path=out2), run_id=s2.run_id).parquet
    assert sha256_file(p1) == s1.checksum
    assert sha256_file(p2) == s2.checksum

    # Strict determinism (byte-identical). Valid in the pinned CI environment
    # and, because the encoder pins thread counts, also in practice locally.
    assert s1.checksum == s2.checksum, (
        "STRICT DETERMINISM FAILED: two batch runs produced different bytes\n"
        f"  run1: {s1.checksum}\n"
        f"  run2: {s2.checksum}"
    )


# --------------------------------------------------------------------------- #
# 3. incremental matches batch over the same catalog
# --------------------------------------------------------------------------- #
def test_incremental_matches_batch(
    tmp_path: pathlib.Path, encoder: OnnxEncoder, onnx_sha: str
) -> None:
    out_batch = tmp_path / "emb_batch"
    out_inc = tmp_path / "emb_inc"

    s_batch = _run(
        catalog=CATALOG, out_dir=out_batch, mode="batch", encoder=encoder, onnx_sha=onnx_sha
    )

    # First incremental: no prior state, encodes everything.
    s_inc1 = _run(
        catalog=CATALOG, out_dir=out_inc, mode="incremental", encoder=encoder, onnx_sha=onnx_sha
    )
    assert s_inc1.encoded == EXPECTED_ROWS

    # Second incremental: state now valid, no changes → no_changes.
    s_inc2 = _run(
        catalog=CATALOG, out_dir=out_inc, mode="incremental", encoder=encoder, onnx_sha=onnx_sha
    )
    assert s_inc2.status == "no_changes"

    # The Parquet bytes from the first incremental run must equal the batch run.
    assert s_batch.checksum is not None
    assert s_inc1.checksum is not None
    assert s_batch.checksum == s_inc1.checksum, (
        "incremental and batch produced different bytes\n"
        f"  batch       = {s_batch.checksum}\n"
        f"  incremental = {s_inc1.checksum}"
    )


# --------------------------------------------------------------------------- #
# 4. no_changes really does not write
# --------------------------------------------------------------------------- #
def test_second_incremental_run_is_a_true_noop(
    tmp_path: pathlib.Path, encoder: OnnxEncoder, onnx_sha: str
) -> None:
    out = tmp_path / "emb"
    s1 = _run(catalog=CATALOG, out_dir=out, mode="incremental", encoder=encoder, onnx_sha=onnx_sha)
    root = art.EmbeddingsRoot(path=out)
    current_file = root.current_file
    mtime_before = current_file.stat().st_mtime_ns
    run_id_before = _read_current(root)

    time.sleep(0.05)  # ensure a difference in mtime would be visible

    s2 = _run(catalog=CATALOG, out_dir=out, mode="incremental", encoder=encoder, onnx_sha=onnx_sha)

    assert s2.status == "no_changes"
    assert s2.run_id == s1.run_id == run_id_before
    assert s2.encoded == 0
    assert s2.unchanged == EXPECTED_ROWS
    # current file untouched
    assert current_file.stat().st_mtime_ns == mtime_before


# --------------------------------------------------------------------------- #
# 5. partial change: exactly the edited item is re-encoded
# --------------------------------------------------------------------------- #
def test_partial_change_encodes_only_changed(
    tmp_path: pathlib.Path, encoder: OnnxEncoder, onnx_sha: str
) -> None:
    out = tmp_path / "emb"
    catalog_v2 = tmp_path / "catalog_v2.jsonl"

    # Seed with batch from the original catalog.
    _run(catalog=CATALOG, out_dir=out, mode="batch", encoder=encoder, onnx_sha=onnx_sha)

    # Build a modified catalog: change exactly one item's description.
    lines = CATALOG.read_text(encoding="utf-8").splitlines()
    modified_index = 0
    original = json.loads(lines[modified_index])
    mutated = {**original, "description": "A completely different description."}
    lines[modified_index] = json.dumps(mutated)
    catalog_v2.write_text("\n".join(lines) + "\n", encoding="utf-8")

    s = _run(
        catalog=catalog_v2, out_dir=out, mode="incremental", encoder=encoder, onnx_sha=onnx_sha
    )
    assert s.status == "ok"
    assert s.total == EXPECTED_ROWS
    assert s.encoded == 1
    assert s.unchanged == EXPECTED_ROWS - 1
    assert s.deleted == 0


# --------------------------------------------------------------------------- #
# 5b. deleted item: counted correctly, and its row is dropped from the output
# --------------------------------------------------------------------------- #
def test_deleted_item_is_counted_and_dropped(
    tmp_path: pathlib.Path, encoder: OnnxEncoder, onnx_sha: str
) -> None:
    out = tmp_path / "emb"
    catalog_v2 = tmp_path / "catalog_v2.jsonl"

    # Seed with batch from the original catalog.
    _run(catalog=CATALOG, out_dir=out, mode="batch", encoder=encoder, onnx_sha=onnx_sha)

    # Build a modified catalog with one item removed.
    lines = CATALOG.read_text(encoding="utf-8").splitlines()
    removed = json.loads(lines[0])
    remaining = lines[1:]
    catalog_v2.write_text("\n".join(remaining) + "\n", encoding="utf-8")

    s = _run(
        catalog=catalog_v2, out_dir=out, mode="incremental", encoder=encoder, onnx_sha=onnx_sha
    )
    assert s.status == "ok"
    assert s.total == EXPECTED_ROWS - 1
    assert s.encoded == 0
    assert s.unchanged == EXPECTED_ROWS - 1
    assert s.deleted == 1

    # And the removed item is not in the new Parquet.
    root = art.EmbeddingsRoot(path=out)
    run_paths = art.RunPaths(root=root, run_id=s.run_id)
    table = pq.read_table(str(run_paths.parquet), columns=["item_id"])
    present = set(table.column("item_id").to_pylist())
    assert removed["item_id"] not in present
    assert len(present) == EXPECTED_ROWS - 1


# --------------------------------------------------------------------------- #
# 6. CLI stdout contract
# --------------------------------------------------------------------------- #
def test_cli_prints_single_json_line_to_stdout(
    tmp_path: pathlib.Path,
    onnx_dir: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "emb"
    rc = pl.main(
        [
            "--mode",
            "batch",
            "--catalog",
            str(CATALOG),
            "--out",
            str(out),
            "--onnx-dir",
            str(onnx_dir),
            "--label",
            LABEL,
            "--quiet",
        ]
    )
    assert rc == pl.EXIT_OK

    captured = capsys.readouterr()
    stdout_lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(stdout_lines) == 1, f"expected 1 stdout line, got {len(stdout_lines)}"
    summary = json.loads(stdout_lines[0])
    assert summary["event"] == "pipeline.done"
    assert summary["status"] == "ok"
    assert summary["mode"] == "batch"
    assert summary["total"] == EXPECTED_ROWS

    # Progress lines (if any) must go to stderr, not stdout.
    assert "pipeline.batch" not in captured.out


# --------------------------------------------------------------------------- #
# 7. lock prevents a concurrent run
# --------------------------------------------------------------------------- #
def test_held_lock_causes_timeout(
    tmp_path: pathlib.Path, encoder: OnnxEncoder, onnx_sha: str
) -> None:
    out = tmp_path / "emb"
    root = art.EmbeddingsRoot(path=out)

    with (
        art.file_lock(root.lock_file, timeout=2.0),
        pytest.raises(art.LockTimeoutError, match="could not acquire"),
    ):
        _run(
            catalog=CATALOG,
            out_dir=out,
            mode="batch",
            encoder=encoder,
            onnx_sha=onnx_sha,
            lock_timeout=0.3,
        )
    # After release, a normal run succeeds.
    s = _run(
        catalog=CATALOG,
        out_dir=out,
        mode="batch",
        encoder=encoder,
        onnx_sha=onnx_sha,
        lock_timeout=2.0,
    )
    assert s.status == "ok"
