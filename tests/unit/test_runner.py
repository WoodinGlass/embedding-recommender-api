"""Unit tests for the evaluation runner.

The runner takes its inputs as arguments, so it can be exercised with
small in-memory backends and a dictionary-backed embedding lookup. No
database, no pyarrow, no ONNX. Runs in Colab as well as CI.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.evaluation.golden_set import GoldenQuery, GoldenSet
from recsys.evaluation.runner import (
    EvaluationError,
    build_report,
    encode_query,
    evaluate_system,
    write_report,
)
from recsys.retrieval.numpy_backend import NumpyBackend

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _unit_row(dim: int, index: int) -> NDArray[np.float32]:
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


def _backend(item_ids: list[str], dim: int = 4) -> NumpyBackend:
    # One-hot vectors: item i is the i-th basis vector. Cosine similarity
    # between an item and its own vector is 1.0; between two distinct items
    # it is 0.0. This makes expected top-k unambiguous.
    vecs = np.stack([_unit_row(dim, i) for i in range(len(item_ids))]).astype(np.float32)
    return NumpyBackend(item_ids=item_ids, embeddings=vecs)


def _golden_set(
    queries: list[GoldenQuery] | None = None,
) -> GoldenSet:
    if queries is None:
        queries = [
            GoldenQuery(
                query_id="q1",
                topic="t",
                seed_item_ids=("a",),
                relevant_item_ids=("b", "c"),
            ),
        ]
    return GoldenSet(
        version="v1",
        path=pathlib.Path("evaluation/golden_set/v1.jsonl"),
        queries=tuple(queries),
    )


def _lookup_from_backend(
    backend: NumpyBackend,
) -> Callable[[str], NDArray[np.float32] | None]:
    """Return a dict-backed embedding lookup for a NumpyBackend."""
    table = {iid: backend._embeddings[i] for i, iid in enumerate(backend.item_ids)}

    def lookup(iid: str) -> NDArray[np.float32] | None:
        return table.get(iid)

    return lookup


# --------------------------------------------------------------------------- #
# encode_query
# --------------------------------------------------------------------------- #
def test_encode_query_single_seed() -> None:
    vecs = {"a": _unit_row(4, 0)}
    q = encode_query(["a"], embedding_lookup=vecs.get)
    assert q.dtype == np.float32
    assert q.shape == (4,)
    np.testing.assert_allclose(q, _unit_row(4, 0), atol=1e-6)


def test_encode_query_mean_then_normalize() -> None:
    # Mean of two orthogonal unit vectors is (0.5, 0.5, 0, 0); normalized
    # that is (1/sqrt2, 1/sqrt2, 0, 0).
    vecs = {"a": _unit_row(4, 0), "b": _unit_row(4, 1)}
    q = encode_query(["a", "b"], embedding_lookup=vecs.get)
    assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=1e-6)
    expected = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32) / np.sqrt(2.0)
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_encode_query_skips_missing_seeds() -> None:
    vecs = {"a": _unit_row(4, 0)}
    q = encode_query(["a", "missing"], embedding_lookup=vecs.get)
    np.testing.assert_allclose(q, _unit_row(4, 0), atol=1e-6)


def test_encode_query_raises_on_empty_seeds() -> None:
    with pytest.raises(EvaluationError, match="no seeds"):
        encode_query([], embedding_lookup=lambda _: None)


def test_encode_query_raises_when_all_missing() -> None:
    with pytest.raises(EvaluationError, match="none of the seed items"):
        encode_query(["a", "b"], embedding_lookup=lambda _: None)


def test_encode_query_raises_on_zero_mean() -> None:
    # Two opposite unit vectors average to zero, which cannot be normalized.
    vecs = {
        "a": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "b": np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    with pytest.raises(EvaluationError, match="zero vector"):
        encode_query(["a", "b"], embedding_lookup=vecs.get)


# --------------------------------------------------------------------------- #
# evaluate_system
# --------------------------------------------------------------------------- #
def test_evaluate_system_perfect_retrieval() -> None:
    # Backend has a, b, c, d. Seed is "a"; relevant is "b". The query
    # vector is a's own vector, so a would rank first without seed
    # exclusion; with exclusion, top-1 is "b" (relevant).
    backend = _backend(["a", "b", "c", "d"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set(
        [
            GoldenQuery(
                query_id="q",
                topic="t",
                seed_item_ids=("a",),
                relevant_item_ids=("b",),
            ),
        ]
    )
    metrics, outcomes = evaluate_system(
        backend=backend, golden_set=gs, embedding_lookup=lookup, k=1
    )
    assert metrics["recall_at_1"] == 1.0
    assert metrics["ndcg_at_1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert outcomes == []


def test_evaluate_system_seed_is_excluded_from_results() -> None:
    """The seed item must never appear in the retrieved results.

    The query vector is the mean of the seed embeddings, so the seeds are
    the nearest neighbours of their own mean. Recommending an item the
    user already has is wrong, and it would artificially depress MRR and
    NDCG by pushing relevant items down.
    """
    backend = _backend(["a", "b", "c"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set([GoldenQuery("q", "t", ("a",), ("b",))])
    _, outcomes = evaluate_system(
        backend=backend,
        golden_set=gs,
        embedding_lookup=lookup,
        k=3,
        include_per_query=True,
    )
    assert "a" not in outcomes[0].retrieved
    assert "b" in outcomes[0].retrieved


def test_evaluate_system_zero_retrieval() -> None:
    # Seed is "a", relevant is "d". After filtering the seed, top-1 is "b";
    # "d" is not retrieved at k=1.
    backend = _backend(["a", "b", "c", "d"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set(
        [
            GoldenQuery(
                query_id="q",
                topic="t",
                seed_item_ids=("a",),
                relevant_item_ids=("d",),
            ),
        ]
    )
    metrics, _ = evaluate_system(backend=backend, golden_set=gs, embedding_lookup=lookup, k=1)
    assert metrics["recall_at_1"] == 0.0
    assert metrics["ndcg_at_1"] == 0.0
    assert metrics["mrr"] == 0.0


def test_evaluate_system_aggregates_over_queries() -> None:
    # Two queries, one hit and one miss at k=1.
    backend = _backend(["a", "b"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set(
        [
            GoldenQuery("q1", "t", ("a",), ("b",)),  # hit
            GoldenQuery("q2", "t", ("a",), ("a",)),  # miss (relevant == seed)
        ]
    )
    # q1: after excluding "a", top-1 is "b" (relevant) -> recall=1.0
    # q2: relevant is {"a"}, seed is also "a"; after filtering, "a" is gone
    #     from the results, so recall=0.0. This pair is chosen only to
    #     exercise the aggregate; the golden set loader would reject it.
    metrics, _ = evaluate_system(backend=backend, golden_set=gs, embedding_lookup=lookup, k=1)
    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["ndcg_at_1"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.5)


def test_evaluate_system_metric_names_include_k() -> None:
    backend = _backend(["a", "b"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set([GoldenQuery("q", "t", ("a",), ("b",))])
    metrics, _ = evaluate_system(backend=backend, golden_set=gs, embedding_lookup=lookup, k=5)
    assert "recall_at_5" in metrics
    assert "ndcg_at_5" in metrics
    assert "mrr" in metrics


def test_evaluate_system_without_exact_has_no_fidelity() -> None:
    backend = _backend(["a", "b"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set([GoldenQuery("q", "t", ("a",), ("b",))])
    metrics, _ = evaluate_system(backend=backend, golden_set=gs, embedding_lookup=lookup, k=1)
    assert "ann_recall_vs_exact" not in metrics


def test_evaluate_system_with_exact_has_perfect_fidelity_when_identical() -> None:
    backend = _backend(["a", "b", "c"])
    exact = _backend(["a", "b", "c"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set([GoldenQuery("q", "t", ("a",), ("b",))])
    metrics, _ = evaluate_system(
        backend=backend,
        golden_set=gs,
        embedding_lookup=lookup,
        k=2,
        exact_backend=exact,
    )
    assert metrics["ann_recall_vs_exact"] == 1.0


def test_evaluate_system_with_exact_measures_zero_overlap() -> None:
    backend = _backend(["a", "b", "c", "d"])

    # A backend that always returns "d", regardless of query or k. The
    # system under test returns [b, c] (top-2 after filtering seed "a");
    # the fixed exact side returns [d] (a is not among its results, so
    # seed filtering is a no-op on it). Overlap empty.
    class _FixedBackend:
        name = "fixed"

        def is_ready(self) -> bool:
            return True

        def search(
            self,
            *,
            vector: NDArray[np.float32],
            k: int,
            filters: object = None,
        ) -> list[tuple[str, float]]:
            del vector, filters, k
            return [("d", 0.9)]

    lookup = _lookup_from_backend(backend)
    gs = _golden_set([GoldenQuery("q", "t", ("a",), ("b",))])
    metrics, _ = evaluate_system(
        backend=backend,
        golden_set=gs,
        embedding_lookup=lookup,
        k=2,
        exact_backend=_FixedBackend(),
    )
    assert metrics["ann_recall_vs_exact"] == 0.0


def test_evaluate_system_with_exact_measures_partial_overlap() -> None:
    backend = _backend(["a", "b", "c", "d"])

    # An exact backend that returns [a, c] ignoring the query. After seed
    # filtering, the exact side is [c]; the system side is [b, c].
    # Overlap is {"c"}, k=2, fidelity = 1/2.
    class _OverlappingBackend:
        name = "overlapping"

        def is_ready(self) -> bool:
            return True

        def search(
            self,
            *,
            vector: NDArray[np.float32],
            k: int,
            filters: object = None,
        ) -> list[tuple[str, float]]:
            del vector, filters, k
            return [("a", 0.9), ("c", 0.8)]

    lookup = _lookup_from_backend(backend)
    gs = _golden_set([GoldenQuery("q", "t", ("a",), ("b",))])
    metrics, _ = evaluate_system(
        backend=backend,
        golden_set=gs,
        embedding_lookup=lookup,
        k=2,
        exact_backend=_OverlappingBackend(),
    )
    assert metrics["ann_recall_vs_exact"] == pytest.approx(0.5)


def test_evaluate_system_rejects_non_positive_k() -> None:
    backend = _backend(["a"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set([GoldenQuery("q", "t", ("a",), ("b",))])
    with pytest.raises(EvaluationError, match="k must be positive"):
        evaluate_system(backend=backend, golden_set=gs, embedding_lookup=lookup, k=0)


def test_evaluate_system_per_query_outcomes_when_requested() -> None:
    backend = _backend(["a", "b"])
    lookup = _lookup_from_backend(backend)
    gs = _golden_set(
        [
            GoldenQuery("q1", "t", ("a",), ("b",)),  # hit
            GoldenQuery("q2", "t", ("a",), ("a",)),  # miss (relevant == seed)
        ]
    )
    _, outcomes = evaluate_system(
        backend=backend,
        golden_set=gs,
        embedding_lookup=lookup,
        k=1,
        include_per_query=True,
    )
    assert len(outcomes) == 2
    assert outcomes[0].query_id == "q1"
    assert outcomes[0].metrics["recall_at_1"] == 1.0
    assert outcomes[1].query_id == "q2"
    assert outcomes[1].metrics["recall_at_1"] == 0.0


# --------------------------------------------------------------------------- #
# build_report
# --------------------------------------------------------------------------- #
def test_build_report_shape() -> None:
    report = build_report(
        golden_set_version="v1",
        commit="abc123",
        created_at="2026-10-08T12:00:00Z",
        systems={
            "pgvector_hnsw": {"recall_at_10": 0.9, "ndcg_at_10": 0.85},
            "random_baseline": {"recall_at_10": 0.1},
        },
    )
    assert report["schema_version"] == 1
    assert report["golden_set_version"] == "v1"
    assert report["commit"] == "abc123"
    assert report["created_at"] == "2026-10-08T12:00:00Z"
    assert report["systems"]["pgvector_hnsw"]["metrics"]["recall_at_10"] == 0.9
    assert report["systems"]["random_baseline"]["metrics"]["recall_at_10"] == 0.1


def test_build_report_with_per_query() -> None:
    outcome = type(
        "O",
        (),
        {
            "query_id": "q1",
            "retrieved": ("a", "b"),
            "metrics": {"recall_at_1": 1.0},
        },
    )()
    report = build_report(
        golden_set_version="v1",
        commit="abc",
        created_at="t",
        systems={"s": {"recall_at_1": 1.0}},
        per_query={"s": [outcome]},
    )
    pq = report["systems"]["s"]["per_query"]
    assert len(pq) == 1
    assert pq[0]["query_id"] == "q1"
    assert pq[0]["retrieved"] == ["a", "b"]


def test_build_report_without_per_query_omits_key() -> None:
    report = build_report(
        golden_set_version="v1",
        commit="abc",
        created_at="t",
        systems={"s": {"recall_at_1": 1.0}},
    )
    assert "per_query" not in report["systems"]["s"]


# --------------------------------------------------------------------------- #
# write_report
# --------------------------------------------------------------------------- #
def test_write_report_roundtrip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "report.json"
    doc = {"schema_version": 1, "commit": "abc"}
    write_report(path, doc)
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == doc


def test_write_report_leaves_no_tmp(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "report.json"
    write_report(path, {"a": 1})
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_write_report_overwrites(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "report.json"
    write_report(path, {"a": 1})
    write_report(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}


def test_write_report_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "nested" / "dir" / "report.json"
    write_report(path, {"a": 1})
    assert path.is_file()
