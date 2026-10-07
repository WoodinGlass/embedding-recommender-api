"""Unit tests for :mod:`recsys.retrieval.build`.

The DB-touching part (``build_index``) is exercised end-to-end in
``tests/integration/test_index_lifecycle.py``. These tests cover the pure
part — resolving a run's inputs from disk, reading a catalog — which needs
no database and runs in Colab as well as CI.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from recsys.embeddings.artifacts import ParquetBatchWriter
from recsys.retrieval.build import (
    BuildInputError,
    BuildInputs,
    read_catalog_items,
    resolve_build_inputs,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_run(
    root: pathlib.Path,
    *,
    run_id: str = "2026-10-08T11-30-00Z__minilm-onnx-v1+a3f9e021",
    item_ids: list[str] | None = None,
    rows: int = 3,
) -> pathlib.Path:
    """Write a minimal run directory with a manifest and a Parquet."""
    if item_ids is None:
        item_ids = [f"i_{i:04d}" for i in range(rows)]
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Parquet
    dim = 4
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((len(item_ids), dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vecs = (raw / norms).astype(np.float32)
    with ParquetBatchWriter(run_dir / "embeddings.parquet", embedding_dim=dim) as w:
        w.write_batch(
            item_ids=list(item_ids),
            embeddings=vecs,
            content_hashes=["h" * 16 for _ in item_ids],
            preprocessing_version="v1",
        )

    # Manifest
    manifest = {
        "schema_version": 1,
        "created_at": "2026-10-08T11:30:00Z",
        "run_id": run_id,
        "mode": "batch",
        "model_version": "minilm-onnx-v1+a3f9e021",
        "preprocessing_version": "v1",
        "config_hash": "sha256:abc",
        "catalog_snapshot": "sha256:9f1e2c8a3b5d7e4f",
        "onnx_artifact_sha256": "a" * 64,
        "parquet": {
            "path": f"runs/{run_id}/embeddings.parquet",
            "sha256": "b" * 64,
            "rows": len(item_ids),
            "encoded_rows": len(item_ids),
            "dim": dim,
            "dtype": "float32",
        },
        "state": {
            "path": f"runs/{run_id}/state.json",
            "sha256": "c" * 64,
        },
        "environment": {
            "python_version": "3.11.0",
            "onnxruntime_version": "1.19.0",
            "numpy_version": "2.1.0",
            "pyarrow_version": "17.0.0",
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # state.json (empty items map is fine for these tests)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": "minilm-onnx-v1+a3f9e021",
                "preprocessing_version": "v1",
                "config_hash": "sha256:abc",
                "hash_algorithm": "sha256",
                "catalog_snapshot": "sha256:9f1e2c8a3b5d7e4f",
                "items": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _write_current(root: pathlib.Path, run_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "current").write_text(run_id + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# resolve_build_inputs — happy path via current
# --------------------------------------------------------------------------- #
def test_resolve_via_current(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "emb"
    run_id = "2026-10-08T11-30-00Z__minilm-onnx-v1+a3f9e021"
    _write_run(root, run_id=run_id)
    _write_current(root, run_id)

    inputs = resolve_build_inputs(root)
    assert isinstance(inputs, BuildInputs)
    assert inputs.run_id == run_id
    assert inputs.model_version == "minilm-onnx-v1+a3f9e021"
    assert inputs.catalog_snapshot == "sha256:9f1e2c8a3b5d7e4f"
    assert inputs.preprocessing_version == "v1"
    assert inputs.onnx_artifact_sha256 == "a" * 64
    assert inputs.parquet_rows == 3
    assert inputs.parquet_path.is_file()
    assert inputs.parquet_path == root / "runs" / run_id / "embeddings.parquet"


# --------------------------------------------------------------------------- #
# resolve_build_inputs — happy path via explicit run_id
# --------------------------------------------------------------------------- #
def test_resolve_via_explicit_run_id(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "emb"
    run_id = "2026-10-08T11-30-00Z__minilm-onnx-v1+a3f9e021"
    _write_run(root, run_id=run_id)
    # No `current` pointer written.

    inputs = resolve_build_inputs(root, run_id=run_id)
    assert inputs.run_id == run_id


# --------------------------------------------------------------------------- #
# resolve_build_inputs — error paths
# --------------------------------------------------------------------------- #
def test_missing_current_pointer(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "emb"
    root.mkdir()
    with pytest.raises(BuildInputError, match="no active run"):
        resolve_build_inputs(root)


def test_empty_current_pointer(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "emb"
    root.mkdir()
    (root / "current").write_text("\n", encoding="utf-8")
    with pytest.raises(BuildInputError, match="current pointer is empty"):
        resolve_build_inputs(root)


def test_missing_manifest(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "emb"
    root.mkdir()
    (root / "current").write_text("run-xyz\n", encoding="utf-8")
    (root / "runs" / "run-xyz").mkdir(parents=True)
    with pytest.raises(BuildInputError, match="manifest not found"):
        resolve_build_inputs(root)


def test_malformed_manifest_json(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "emb"
    run_dir = root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (root / "current").write_text("run-1\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(BuildInputError, match="not valid JSON"):
        resolve_build_inputs(root)


def test_manifest_missing_field(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "emb"
    run_dir = root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (root / "current").write_text("run-1\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({"model_version": "m"}), encoding="utf-8")
    with pytest.raises(BuildInputError, match="missing field"):
        resolve_build_inputs(root)


def test_parquet_missing(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "emb"
    run_dir = root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (root / "current").write_text("run-1\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": "m",
                "catalog_snapshot": "s",
                "preprocessing_version": "v1",
                "onnx_artifact_sha256": "a" * 64,
                "parquet": {"path": "runs/run-1/embeddings.parquet", "rows": 0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BuildInputError, match="parquet not found"):
        resolve_build_inputs(root)


# --------------------------------------------------------------------------- #
# read_catalog_items
# --------------------------------------------------------------------------- #
def _write_catalog(path: pathlib.Path, items: list[dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")


def _item(iid: str) -> dict[str, str]:
    return {
        "item_id": iid,
        "title": f"Title {iid}",
        "description": f"Description {iid}",
        "category": "books",
        "brand": "ace",
    }


def test_read_catalog_items_happy_path(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_catalog(p, [_item("i_0001"), _item("i_0002")])
    items = read_catalog_items(p)
    assert set(items.keys()) == {"i_0001", "i_0002"}
    assert items["i_0001"]["title"] == "Title i_0001"


def test_read_catalog_items_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(BuildInputError, match="catalog not found"):
        read_catalog_items(tmp_path / "nope.jsonl")


def test_read_catalog_items_empty(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(BuildInputError, match="contains no items"):
        read_catalog_items(p)


def test_read_catalog_items_duplicate_id(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_catalog(p, [_item("i_0001"), _item("i_0001")])
    with pytest.raises(BuildInputError, match="duplicate item_id"):
        read_catalog_items(p)


def test_read_catalog_items_missing_field(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"item_id": "i_0001"}) + "\n", encoding="utf-8")
    with pytest.raises(BuildInputError, match="missing fields"):
        read_catalog_items(p)


def test_read_catalog_items_invalid_json(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text("{ not json\n", encoding="utf-8")
    with pytest.raises(BuildInputError, match="invalid JSON"):
        read_catalog_items(p)


def test_read_catalog_items_skips_blank_lines(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text("\n" + json.dumps(_item("i_0001")) + "\n\n", encoding="utf-8")
    items = read_catalog_items(p)
    assert list(items.keys()) == ["i_0001"]
