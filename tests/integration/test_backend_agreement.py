"""Backend agreement: NumpyBackend and PgvectorBackend on exact search.

ADR-0012 requires the two backends to agree on what "top-k by cosine
distance" means. With HNSW's ``ef_search`` large enough that the graph
walk degenerates to a full scan, the pgvector backend must return the
same neighbours, in the same order, with scores within a small tolerance
of the numpy backend.

This is the regression test that keeps the ANN-fidelity metric (ADR-0009)
meaningful: if the two backends disagree on the ground truth, then a
lower-than-1.0 fidelity score would measure nothing.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.retrieval.build import (
    build_index,
    read_catalog_items,
    resolve_build_inputs,
)
from recsys.retrieval.numpy_backend import NumpyBackend
from recsys.retrieval.pgvector import PgvectorBackend
from recsys.retrieval.promote import promote

pytestmark = pytest.mark.integration

# Must match the fixed dimension of the embedding.vector column
# (ADR-0006): vector(384) for the pinned all-MiniLM-L6-v2 model.
EMBEDDING_DIM: int = 384

TOLERANCE = 1e-5


# --------------------------------------------------------------------------- #
# fixtures (same synthetic run shape as test_index_lifecycle)
# --------------------------------------------------------------------------- #
def _write_run(
    root: pathlib.Path,
    *,
    run_id: str,
    item_ids: list[str],
    seed: int,
) -> NDArray[np.float32]:
    """Write a run directory and return its embeddings for cross-check."""
    from recsys.embeddings.artifacts import ParquetBatchWriter

    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dim = EMBEDDING_DIM
    rng = np.random.default_rng(seed)
    raw: NDArray[np.float32] = rng.standard_normal((len(item_ids), dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vecs = cast(NDArray[np.float32], (raw / norms).astype(np.float32))

    with ParquetBatchWriter(run_dir / "embeddings.parquet", embedding_dim=dim) as w:
        w.write_batch(
            item_ids=list(item_ids),
            embeddings=vecs,
            content_hashes=["h" * 16 for _ in item_ids],
            preprocessing_version="v1",
        )

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
        "state": {"path": f"runs/{run_id}/state.json", "sha256": "c" * 64},
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
    (run_dir / "state.json").write_text("{}\n", encoding="utf-8")
    (root / "current").write_text(run_id + "\n", encoding="utf-8")
    return vecs


def _write_catalog(path: pathlib.Path, item_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "item_id": iid,
                "title": f"Title {iid}",
                "description": f"Description {iid}",
                "category": "books",
                "brand": "ace",
            }
        )
        for iid in item_ids
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def shared_run(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, list[str], NDArray[np.float32]]:
    """One run shared by both backends: same vectors, same order."""
    item_ids = [f"i_{i:04d}" for i in range(20)]
    root = tmp_path / "emb"
    vecs = _write_run(
        root,
        run_id="2026-10-08T11-30-00Z__minilm-onnx-v1+a3f9e021",
        item_ids=item_ids,
        seed=42,
    )
    return root, item_ids, vecs


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_backends_agree_on_exact_search(
    clean_db: None,
    pg_connection: Any,
    shared_run: tuple[pathlib.Path, list[str], NDArray[np.float32]],
    tmp_path: pathlib.Path,
) -> None:
    root, item_ids, _vecs = shared_run

    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog, item_ids)

    # Build and promote through the real lifecycle, so pgvector ends up
    # with the same rows the numpy backend reads from Parquet.
    inputs = resolve_build_inputs(root)
    items = read_catalog_items(catalog)
    built = build_index(
        pg_connection,
        inputs=inputs,
        catalog_items=items,
        metric="cosine",
        hnsw_m=16,
        hnsw_ef_construction=64,
        hnsw_ef_search=100,
    )
    promote(pg_connection, target_index_version=built.index_version)

    numpy_backend = NumpyBackend.from_run_directory(inputs.run_dir)
    # ef_search >= number of rows makes HNSW degenerate to exact search.
    pg_backend = PgvectorBackend.from_registry(
        pg_connection,
        hnsw_ef_search=len(item_ids) + 1,
    )

    # Query with a vector not in the catalog (a fresh point), so the answer
    # is not trivially "the same row".
    rng = np.random.default_rng(1234)
    q = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    q = (q / np.linalg.norm(q)).astype(np.float32)

    numpy_results = numpy_backend.search(vector=q, k=10)
    pg_results = pg_backend.search(vector=q, k=10)

    numpy_ids = [iid for iid, _ in numpy_results]
    pg_ids = [iid for iid, _ in pg_results]
    assert numpy_ids == pg_ids, (
        f"backends disagree on top-10 by cosine distance\n  numpy: {numpy_ids}\n  pg   : {pg_ids}"
    )

    numpy_scores = np.array([s for _, s in numpy_results])
    pg_scores = np.array([s for _, s in pg_results])
    np.testing.assert_allclose(
        numpy_scores,
        pg_scores,
        atol=TOLERANCE,
        err_msg="scores differ beyond tolerance",
    )


def test_backends_agree_with_filter(
    clean_db: None,
    pg_connection: Any,
    shared_run: tuple[pathlib.Path, list[str], NDArray[np.float32]],
    tmp_path: pathlib.Path,
) -> None:
    root, item_ids, _ = shared_run
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog, item_ids)

    inputs = resolve_build_inputs(root)
    items = read_catalog_items(catalog)
    built = build_index(
        pg_connection,
        inputs=inputs,
        catalog_items=items,
        metric="cosine",
        hnsw_m=16,
        hnsw_ef_construction=64,
        hnsw_ef_search=100,
    )
    promote(pg_connection, target_index_version=built.index_version)

    # Metadata for the numpy backend: same category for every item, so the
    # filter is "match everything" and the two must still agree.
    metadata = {iid: {"category": "books", "brand": "ace", "language": "en"} for iid in item_ids}
    numpy_backend = NumpyBackend.from_run_directory(inputs.run_dir, item_metadata=metadata)
    pg_backend = PgvectorBackend.from_registry(pg_connection, hnsw_ef_search=len(item_ids) + 1)

    rng = np.random.default_rng(99)
    q = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    q = (q / np.linalg.norm(q)).astype(np.float32)

    filters = {"category": "books"}
    numpy_results = numpy_backend.search(vector=q, k=10, filters=filters)
    pg_results = pg_backend.search(vector=q, k=10, filters=filters)

    assert [iid for iid, _ in numpy_results] == [iid for iid, _ in pg_results]
