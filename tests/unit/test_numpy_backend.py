"""Unit tests for :class:`NumpyBackend`.

The backend is in-process and depends on nothing but numpy and pyarrow,
so the whole contract can be exercised here, on any machine, without a
database. The integration test that compares it against ``PgvectorBackend``
lives in ``tests/integration/test_backend_agreement.py`` (M2.2).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.embeddings.artifacts import ParquetBatchWriter
from recsys.retrieval.base import IndexBackend
from recsys.retrieval.numpy_backend import NumpyBackend

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _unit_vector(dim: int, index: int) -> NDArray[np.float32]:
    """One-hot unit vector: the ``index``-th basis vector of dimension ``dim``."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


def _fixture() -> tuple[list[str], NDArray[np.float32], dict[str, dict[str, str]]]:
    """Four orthogonal unit vectors in 4-D, with metadata.

    The four vectors are mutually orthogonal, so each is the exact nearest
    neighbor of itself with cosine similarity 1.0 and every other vector
    has similarity 0.0. That makes expected top-k unambiguous.
    """
    ids = ["i_0003", "i_0001", "i_0004", "i_0002"]  # deliberately unsorted
    vecs = np.stack([_unit_vector(4, i) for i in range(4)]).astype(np.float32)
    metadata = {
        "i_0001": {"category": "books", "brand": "ace", "language": "en"},
        "i_0002": {"category": "books", "brand": "tor", "language": "en"},
        "i_0003": {"category": "music", "brand": "blue_note", "language": "en"},
        "i_0004": {"category": "music", "brand": "verve", "language": "fr"},
    }
    return ids, vecs, metadata


# --------------------------------------------------------------------------- #
# construction validation
# --------------------------------------------------------------------------- #
def test_construction_happy_path() -> None:
    ids, vecs, meta = _fixture()
    be = NumpyBackend(item_ids=ids, embeddings=vecs, item_metadata=meta)
    assert be.size == 4
    assert be.embedding_dim == 4
    assert be.name == "numpy"


def test_construction_sorts_internally_by_item_id() -> None:
    ids, vecs, meta = _fixture()
    be = NumpyBackend(item_ids=ids, embeddings=vecs, item_metadata=meta)
    assert be.item_ids == ["i_0001", "i_0002", "i_0003", "i_0004"]


def test_construction_rejects_length_mismatch() -> None:
    ids, vecs, meta = _fixture()
    with pytest.raises(ValueError, match="entries but embeddings has"):
        NumpyBackend(item_ids=ids[:3], embeddings=vecs, item_metadata=meta)


def test_construction_rejects_1d_embeddings() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        NumpyBackend(
            item_ids=["a", "b"],
            embeddings=np.zeros(2, dtype=np.float32),
        )


def test_construction_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="must be float32"):
        NumpyBackend(
            item_ids=["a", "b"],
            embeddings=np.zeros((2, 4), dtype=np.float64),
        )


def test_construction_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        NumpyBackend(
            item_ids=["a", "a"],
            embeddings=np.eye(2, dtype=np.float32),
        )


# --------------------------------------------------------------------------- #
# protocol conformance
# --------------------------------------------------------------------------- #
def test_satisfies_index_backend_protocol() -> None:
    ids, vecs, meta = _fixture()
    be = NumpyBackend(item_ids=ids, embeddings=vecs, item_metadata=meta)
    assert isinstance(be, IndexBackend)


def test_is_ready_always_true() -> None:
    ids, vecs, meta = _fixture()
    be = NumpyBackend(item_ids=ids, embeddings=vecs, item_metadata=meta)
    assert be.is_ready() is True


def test_embedding_for_returns_stored_vector() -> None:
    be = _backend()
    v = be.embedding_for("i_0001")
    assert v is not None
    assert v.shape == (4,)
    # Round trip: querying with the returned vector makes i_0001 the top hit.
    results = be.search(vector=v, k=1)
    assert results[0][0] == "i_0001"
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_embedding_for_returns_none_for_unknown_id() -> None:
    be = _backend()
    assert be.embedding_for("no-such-id") is None


# --------------------------------------------------------------------------- #
# search: basic behavior
# --------------------------------------------------------------------------- #
def _backend() -> NumpyBackend:
    ids, vecs, meta = _fixture()
    return NumpyBackend(item_ids=ids, embeddings=vecs, item_metadata=meta)


def test_search_returns_exact_match_first() -> None:
    be = _backend()
    q = _unit_vector(4, 1)  # i_0001's vector
    results = be.search(vector=q, k=4)
    assert results[0][0] == "i_0001"
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_search_scores_are_clamped_to_unit_interval() -> None:
    be = _backend()
    q = _unit_vector(4, 0)
    results = be.search(vector=q, k=4)
    for _, score in results:
        assert 0.0 <= score <= 1.0


def test_search_respects_k() -> None:
    be = _backend()
    q = _unit_vector(4, 0)
    assert len(be.search(vector=q, k=1)) == 1
    assert len(be.search(vector=q, k=2)) == 2
    assert len(be.search(vector=q, k=4)) == 4


def test_search_k_larger_than_catalog_returns_all() -> None:
    be = _backend()
    q = _unit_vector(4, 0)
    assert len(be.search(vector=q, k=100)) == 4


def test_search_sorted_by_score_descending() -> None:
    be = _backend()
    # A query that is not a basis vector: has nonzero similarity to two
    # items and zero to the rest. Verify the ordering.
    q = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    results = be.search(vector=q, k=4)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_search_tie_break_is_item_id_ascending() -> None:
    """Equal scores resolve to item_id ascending.

    Two items are placed at exactly the same cosine distance from the
    query so their scores are bit-identical. The backend's internal order
    (item_id ascending) must surface as the order among the tie.
    """
    # Two identical vectors for two different item ids.
    ids = ["b", "a"]
    vecs = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    be = NumpyBackend(item_ids=ids, embeddings=vecs)
    q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    results = be.search(vector=q, k=2)
    assert [r[0] for r in results] == ["a", "b"]
    assert results[0][1] == results[1][1]


# --------------------------------------------------------------------------- #
# search: validation
# --------------------------------------------------------------------------- #
def test_search_rejects_non_positive_k() -> None:
    be = _backend()
    with pytest.raises(ValueError, match="k must be positive"):
        be.search(vector=_unit_vector(4, 0), k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        be.search(vector=_unit_vector(4, 0), k=-1)


def test_search_rejects_2d_query() -> None:
    be = _backend()
    with pytest.raises(ValueError, match="must be 1-D"):
        be.search(vector=np.zeros((1, 4), dtype=np.float32), k=1)


def test_search_rejects_wrong_dimension() -> None:
    be = _backend()
    with pytest.raises(ValueError, match="dimension"):
        be.search(vector=_unit_vector(5, 0), k=1)


def test_search_rejects_non_normalized_query() -> None:
    be = _backend()
    q = np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="L2-normalized"):
        be.search(vector=q, k=1)


# --------------------------------------------------------------------------- #
# search: filters
# --------------------------------------------------------------------------- #
def test_filter_category() -> None:
    be = _backend()
    q = _unit_vector(4, 1)
    results = be.search(vector=q, k=10, filters={"category": "books"})
    assert {r[0] for r in results} == {"i_0001", "i_0002"}


def test_filter_language() -> None:
    be = _backend()
    q = _unit_vector(4, 1)
    results = be.search(vector=q, k=10, filters={"language": "fr"})
    assert [r[0] for r in results] == ["i_0004"]


def test_filter_multiple_fields_is_conjunction() -> None:
    be = _backend()
    q = _unit_vector(4, 1)
    results = be.search(vector=q, k=10, filters={"category": "books", "brand": "ace"})
    assert [r[0] for r in results] == ["i_0001"]


def test_filter_matching_nothing_returns_empty() -> None:
    be = _backend()
    q = _unit_vector(4, 1)
    results = be.search(vector=q, k=10, filters={"category": "books", "language": "fr"})
    assert results == []


def test_empty_filter_same_as_no_filter() -> None:
    be = _backend()
    q = _unit_vector(4, 1)
    assert be.search(vector=q, k=4) == be.search(vector=q, k=4, filters=None)
    assert be.search(vector=q, k=4) == be.search(vector=q, k=4, filters={})


def test_filter_with_only_empty_values_same_as_no_filter() -> None:
    be = _backend()
    q = _unit_vector(4, 1)
    assert be.search(vector=q, k=4) == be.search(vector=q, k=4, filters={"category": ""})


def test_unknown_filter_field_raises() -> None:
    be = _backend()
    q = _unit_vector(4, 1)
    with pytest.raises(ValueError, match="unknown filter field"):
        be.search(vector=q, k=4, filters={"color": "red"})


def test_filter_without_metadata_raises() -> None:
    ids, vecs, _ = _fixture()
    be = NumpyBackend(item_ids=ids, embeddings=vecs)  # no metadata
    q = _unit_vector(4, 1)
    with pytest.raises(RuntimeError, match="item_metadata"):
        be.search(vector=q, k=4, filters={"category": "books"})


def test_filter_field_not_in_metadata_raises() -> None:
    ids, vecs, _ = _fixture()
    # Metadata present but missing the 'language' field entirely.
    meta = {
        "i_0001": {"category": "books"},
        "i_0002": {"category": "books"},
        "i_0003": {"category": "music"},
        "i_0004": {"category": "music"},
    }
    be = NumpyBackend(item_ids=ids, embeddings=vecs, item_metadata=meta)
    q = _unit_vector(4, 1)
    with pytest.raises(ValueError, match="not present in item metadata"):
        be.search(vector=q, k=4, filters={"language": "en"})


# --------------------------------------------------------------------------- #
# from_run_directory (uses pyarrow)
# --------------------------------------------------------------------------- #
def test_from_run_directory_reads_parquet(tmp_path: pathlib.Path) -> None:
    ids, vecs, _ = _fixture()
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    parquet = run_dir / "embeddings.parquet"
    with ParquetBatchWriter(parquet, embedding_dim=4) as writer:
        writer.write_batch(
            item_ids=ids,
            embeddings=vecs,
            content_hashes=["h" * 16 for _ in ids],
            preprocessing_version="v1",
        )

    be = NumpyBackend.from_run_directory(run_dir)
    assert be.size == 4
    assert be.embedding_dim == 4
    assert be.item_ids == ["i_0001", "i_0002", "i_0003", "i_0004"]

    q = _unit_vector(4, 1)
    results = be.search(vector=q, k=4)
    assert results[0][0] == "i_0001"


def test_from_run_directory_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"embeddings\.parquet"):
        NumpyBackend.from_run_directory(tmp_path / "nonexistent")


def test_from_run_directory_with_batch_size_smaller_than_catalog(
    tmp_path: pathlib.Path,
) -> None:
    # 200 rows written in batches of 64, read back in batches of 64:
    # concatenation across batches must preserve alignment.
    ids = [f"i_{i:04d}" for i in range(200)]
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((200, 4)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vecs = (raw / norms).astype(np.float32)

    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    with ParquetBatchWriter(run_dir / "embeddings.parquet", embedding_dim=4) as w:
        for start in range(0, 200, 64):
            chunk_ids = ids[start : start + 64]
            chunk = vecs[start : start + 64]
            w.write_batch(
                item_ids=chunk_ids,
                embeddings=chunk,
                content_hashes=["h" * 16 for _ in chunk_ids],
                preprocessing_version="v1",
            )

    be = NumpyBackend.from_run_directory(run_dir)
    assert be.size == 200
    # Query with i_0000's own vector: it must rank first.
    q = vecs[0]
    results = be.search(vector=q, k=1)
    assert results[0][0] == "i_0000"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
