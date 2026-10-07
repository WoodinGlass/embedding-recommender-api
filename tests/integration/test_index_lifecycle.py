"""End-to-end index lifecycle — integration, database tier.

Exercises the M2 build/promote/rollback sequence against a real
PostgreSQL database with the pgvector extension. Marked ``integration``
and skips when ``RECSYS_TEST_DATABASE_URL`` is not set (see
``tests/integration/conftest.py``).

The embeddings are synthetic: a small Parquet run is written on the fly,
bypassing the ONNX encoder. This keeps the test focused on the database
contract (schema, foreign keys, advisory lock, state machine) and fast
enough to run on every PR. The end-to-end encoder path is already
exercised by the M1 pipeline integration suite.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from recsys.retrieval.build import (
    build_index,
    read_catalog_items,
    resolve_build_inputs,
)
from recsys.retrieval.promote import PromoteError, promote, rollback

pytestmark = pytest.mark.integration

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_parquet_run(
    root: pathlib.Path,
    *,
    run_id: str,
    item_ids: list[str],
    model_version: str = "minilm-onnx-v1+a3f9e021",
    catalog_snapshot: str = "sha256:9f1e2c8a3b5d7e4f",
) -> None:
    """Write a minimal but well-formed run directory."""
    from recsys.embeddings.artifacts import ParquetBatchWriter

    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dim = 4
    rng = np.random.default_rng(hash(run_id) % (2**32))
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

    manifest = {
        "schema_version": 1,
        "created_at": "2026-10-08T11:30:00Z",
        "run_id": run_id,
        "mode": "batch",
        "model_version": model_version,
        "preprocessing_version": "v1",
        "config_hash": "sha256:abc",
        "catalog_snapshot": catalog_snapshot,
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
    (run_dir / "state.json").write_text("{}\n", encoding="utf-8")
    (root / "current").write_text(run_id + "\n", encoding="utf-8")


def _write_catalog(path: pathlib.Path, item_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for iid in item_ids:
        lines.append(
            json.dumps(
                {
                    "item_id": iid,
                    "title": f"Title {iid}",
                    "description": f"Description {iid}",
                    "category": "books",
                    "brand": "ace",
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build(
    connection: object,
    *,
    root: pathlib.Path,
    catalog: pathlib.Path,
    hnsw_ef_search: int = 100,
) -> object:
    inputs = resolve_build_inputs(root)
    items = read_catalog_items(catalog)
    return build_index(
        connection,
        inputs=inputs,
        catalog_items=items,
        metric="cosine",
        hnsw_m=16,
        hnsw_ef_construction=64,
        hnsw_ef_search=hnsw_ef_search,
    )


def _active_version(connection: object) -> str | None:
    with connection.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("SELECT index_version FROM index_registry WHERE status = 'active'")
        row = cur.fetchone()
    return None if row is None else str(row[0])


def _status(connection: object, version: str) -> str:
    with connection.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(
            "SELECT status FROM index_registry WHERE index_version = %s",
            (version,),
        )
        row = cur.fetchone()
    assert row is not None, f"index {version!r} not found"
    return str(row[0])


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_run(tmp_path: pathlib.Path) -> pathlib.Path:
    """A single run directory with a small, synthetic embedding set."""
    root = tmp_path / "embeddings"
    _write_parquet_run(
        root,
        run_id="2026-10-08T11-30-00Z__minilm-onnx-v1+a3f9e021",
        item_ids=["i_0001", "i_0002", "i_0003", "i_0004"],
    )
    return root


@pytest.fixture
def sample_catalog(tmp_path: pathlib.Path) -> pathlib.Path:
    catalog = tmp_path / "catalog.jsonl"
    _write_catalog(catalog, ["i_0001", "i_0002", "i_0003", "i_0004"])
    return catalog


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def test_build_creates_building_row(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
) -> None:
    result = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    assert result.status == "created"
    assert result.row_count == 4
    assert _status(pg_connection, result.index_version) == "building"
    # No active index yet.
    assert _active_version(pg_connection) is None


def test_build_is_idempotent(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
) -> None:
    first = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    second = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    assert second.status == "duplicate"
    assert second.index_version == first.index_version

    # Only one registry row.
    with pg_connection.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("SELECT count(*) FROM index_registry")
        count = int(cur.fetchone()[0])  # type: ignore[index]
    assert count == 1


def test_build_populates_item_and_embedding_rows(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
) -> None:
    result = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    with pg_connection.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("SELECT count(*) FROM item")
        items = int(cur.fetchone()[0])  # type: ignore[index]
        cur.execute(
            "SELECT count(*) FROM embedding WHERE index_version = %s",
            (result.index_version,),
        )
        embeddings = int(cur.fetchone()[0])  # type: ignore[index]
    assert items == 4
    assert embeddings == 4


# --------------------------------------------------------------------------- #
# promote
# --------------------------------------------------------------------------- #
def test_promote_activates_building_index(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
) -> None:
    built = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    result = promote(pg_connection, target_index_version=built.index_version)

    assert result.status == "promoted"
    assert result.previous_index_version is None
    assert _status(pg_connection, built.index_version) == "active"
    assert _active_version(pg_connection) == built.index_version


def test_promote_retires_previous_active(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    # First index.
    first = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    promote(pg_connection, target_index_version=first.index_version)

    # Second index with a different catalog snapshot (so a different id).
    root2 = tmp_path / "emb2"
    _write_parquet_run(
        root2,
        run_id="2026-10-08T12-30-00Z__minilm-onnx-v1+a3f9e021",
        item_ids=["i_0001", "i_0002", "i_0003", "i_0004"],
        catalog_snapshot="sha256:0000000000000000",
    )
    second = _build(pg_connection, root=root2, catalog=sample_catalog)
    assert second.index_version != first.index_version

    promote(pg_connection, target_index_version=second.index_version)
    assert _status(pg_connection, first.index_version) == "retired"
    assert _status(pg_connection, second.index_version) == "active"
    assert _active_version(pg_connection) == second.index_version


def test_promote_already_active_is_noop(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
) -> None:
    built = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    promote(pg_connection, target_index_version=built.index_version)
    again = promote(pg_connection, target_index_version=built.index_version)
    assert again.status == "already_active"


def test_promote_rejects_unknown_index(clean_db: None, pg_connection: object) -> None:
    with pytest.raises(PromoteError, match="not found"):
        promote(pg_connection, target_index_version="idx-doesnotexist")


def test_promote_rejects_incomplete_build(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
) -> None:
    built = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    # Simulate a build that recorded the wrong row_count.
    with pg_connection.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute(
            "UPDATE index_registry SET row_count = 999 WHERE index_version = %s",
            (built.index_version,),
        )
    pg_connection.commit()  # type: ignore[attr-defined]

    with pytest.raises(PromoteError, match="row_count=999"):
        promote(pg_connection, target_index_version=built.index_version)


# --------------------------------------------------------------------------- #
# rollback
# --------------------------------------------------------------------------- #
def test_rollback_reactivates_previous(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    first = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    promote(pg_connection, target_index_version=first.index_version)

    root2 = tmp_path / "emb2"
    _write_parquet_run(
        root2,
        run_id="2026-10-08T12-30-00Z__minilm-onnx-v1+a3f9e021",
        item_ids=["i_0001", "i_0002", "i_0003", "i_0004"],
        catalog_snapshot="sha256:0000000000000000",
    )
    second = _build(pg_connection, root=root2, catalog=sample_catalog)
    promote(pg_connection, target_index_version=second.index_version)

    rollback(pg_connection)
    assert _status(pg_connection, first.index_version) == "active"
    assert _status(pg_connection, second.index_version) == "retired"


def test_rollback_without_retired_index_fails(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
) -> None:
    built = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    promote(pg_connection, target_index_version=built.index_version)
    # Nothing retired yet.
    with pytest.raises(PromoteError, match="no retired index"):
        rollback(pg_connection)


# --------------------------------------------------------------------------- #
# invariant: at most one active at all times
# --------------------------------------------------------------------------- #
def test_at_most_one_active_enforced_by_database(
    clean_db: None,
    pg_connection: object,
    sample_run: pathlib.Path,
    sample_catalog: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    first = _build(pg_connection, root=sample_run, catalog=sample_catalog)
    promote(pg_connection, target_index_version=first.index_version)

    root2 = tmp_path / "emb2"
    _write_parquet_run(
        root2,
        run_id="2026-10-08T12-30-00Z__minilm-onnx-v1+a3f9e021",
        item_ids=["i_0001", "i_0002", "i_0003", "i_0004"],
        catalog_snapshot="sha256:0000000000000000",
    )
    second = _build(pg_connection, root=root2, catalog=sample_catalog)

    # Attempt to activate the second without retiring the first. The
    # partial unique index must refuse.
    import psycopg

    def _try_activate() -> None:
        with pg_connection.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute(
                "UPDATE index_registry SET status = 'active' WHERE index_version = %s",
                (second.index_version,),
            )
        pg_connection.commit()  # type: ignore[attr-defined]

    with pytest.raises(psycopg.errors.UniqueViolation):
        _try_activate()
    pg_connection.rollback()  # type: ignore[attr-defined]
