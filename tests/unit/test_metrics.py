"""Unit tests for retrieval metrics.

The definitions are fixed by ADR-0009 § 2. These tests pin every
convention that could plausibly be implemented differently: the recall
denominator, the NDCG log base and offset, the empty-input cases, and the
behavior of ANN fidelity when the two rankings have different lengths.
"""

from __future__ import annotations

import math

import pytest

from recsys.evaluation.metrics import (
    ann_recall_vs_exact,
    mean,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #
def test_recall_perfect() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_recall_zero() -> None:
    assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0


def test_recall_partial() -> None:
    # 2 of 4 relevant items found in top-4.
    assert recall_at_k(["a", "b", "x", "y"], {"a", "b", "c", "d"}, k=4) == 0.5


def test_recall_only_top_k_counts() -> None:
    # "c" is at position 3; k=2 excludes it.
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_recall_denominator_is_min_relevant_k_when_relevant_exceeds_k() -> None:
    # 10 relevant, k=3, all three retrieved are relevant: recall = 3/3 = 1.0,
    # not 3/10.
    relevant = {f"i_{i}" for i in range(10)}
    retrieved = ["i_0", "i_1", "i_2"]
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_denominator_is_relevant_when_relevant_below_k() -> None:
    # 2 relevant, k=10, both retrieved: recall = 2/2 = 1.0.
    assert recall_at_k(["a", "b"], {"a", "b"}, k=10) == 1.0


def test_recall_empty_relevant_is_one() -> None:
    # A query with no ground truth cannot be missed.
    assert recall_at_k(["a", "b"], set(), k=2) == 1.0


def test_recall_non_positive_k() -> None:
    assert recall_at_k(["a"], {"a"}, k=0) == 0.0
    assert recall_at_k(["a"], {"a"}, k=-1) == 0.0


# --------------------------------------------------------------------------- #
# ndcg_at_k
# --------------------------------------------------------------------------- #
def test_ndcg_perfect_at_rank_1() -> None:
    # Single relevant item at position 0: DCG = 1/log2(2) = 1; IDCG = 1.
    assert ndcg_at_k(["a"], {"a"}, k=1) == 1.0


def test_ndcg_perfect_multiple() -> None:
    # Two relevant items first: DCG = 1 + 1/log2(3); IDCG = same.
    val = ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3)
    assert val == pytest.approx(1.0)


def test_ndcg_discounts_lower_ranks() -> None:
    # Relevant item at position 0 vs position 1.
    first = ndcg_at_k(["a", "x"], {"a"}, k=2)
    second = ndcg_at_k(["x", "a"], {"a"}, k=2)
    assert first > second


def test_ndcg_zero_when_no_hits() -> None:
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_ndcg_ideal_uses_min_relevant_k() -> None:
    # 10 relevant items, k=2: the ideal top-2 has both relevant items.
    relevant = {f"i_{i}" for i in range(10)}
    retrieved = ["i_0", "i_1"]
    assert ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)


def test_ndcg_empty_relevant() -> None:
    assert ndcg_at_k(["a"], set(), k=1) == 0.0


def test_ndcg_non_positive_k() -> None:
    assert ndcg_at_k(["a"], {"a"}, k=0) == 0.0
    assert ndcg_at_k(["a"], {"a"}, k=-1) == 0.0


def test_ndcg_matches_hand_computed_value() -> None:
    # retrieved = [r, x, r], relevant = {r, r2}
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG (ideal) = 1/log2(2) + 1/log2(3) = 1 + 0.63092975357...
    # NDCG = 1.5 / (1 + 1/log2(3))
    retrieved = ["r1", "x", "r2"]
    relevant = {"r1", "r2"}
    expected_dcg = 1 / math.log2(2) + 1 / math.log2(4)
    expected_idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected_dcg / expected_idcg)


def test_ndcg_respects_k_truncation() -> None:
    # The only relevant item is at rank 2. With k=1 it is excluded and the
    # score is 0; with k=2 it is found and the score is positive.
    score_k1 = ndcg_at_k(["x", "a"], {"a"}, k=1)
    score_k2 = ndcg_at_k(["x", "a"], {"a"}, k=2)
    assert score_k1 == 0.0
    assert score_k2 > 0.0


def test_ndcg_truncation_at_rank_one_is_perfect() -> None:
    # A relevant item at rank 1 makes NDCG@1 = 1.0 regardless of what
    # comes after it. This is the property the previous version of this
    # test got wrong: the "truncation is worse" intuition only holds when
    # the truncated item is relevant.
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=1) == 1.0


# --------------------------------------------------------------------------- #
# mrr
# --------------------------------------------------------------------------- #
def test_mrr_first_hit_at_rank_1() -> None:
    assert mrr(["a", "b"], {"a"}) == 1.0


def test_mrr_first_hit_at_rank_3() -> None:
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)


def test_mrr_no_hit() -> None:
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_mrr_empty_relevant() -> None:
    assert mrr(["a"], set()) == 0.0


def test_mrr_empty_retrieved() -> None:
    assert mrr([], {"a"}) == 0.0


def test_mrr_uses_first_hit_not_best() -> None:
    # Both a and b relevant; first is at rank 2.
    assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(1 / 2)


# --------------------------------------------------------------------------- #
# ann_recall_vs_exact
# --------------------------------------------------------------------------- #
def test_ann_fidelity_perfect() -> None:
    assert ann_recall_vs_exact(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0


def test_ann_fidelity_partial() -> None:
    # 2 of 3 overlap.
    assert ann_recall_vs_exact(["a", "b", "x"], ["a", "b", "c"], k=3) == pytest.approx(2 / 3)


def test_ann_fidelity_zero_overlap() -> None:
    assert ann_recall_vs_exact(["x", "y", "z"], ["a", "b", "c"], k=3) == 0.0


def test_ann_fidelity_both_empty_is_one() -> None:
    assert ann_recall_vs_exact([], [], k=3) == 1.0


def test_ann_fidelity_one_empty_is_zero() -> None:
    assert ann_recall_vs_exact(["a", "b"], [], k=2) == 0.0


def test_ann_fidelity_non_positive_k() -> None:
    assert ann_recall_vs_exact(["a"], ["a"], k=0) == 0.0


def test_ann_fidelity_ignores_order() -> None:
    # Fidelity is a set-overlap metric, not a ranking metric.
    assert ann_recall_vs_exact(["a", "b", "c"], ["c", "b", "a"], k=3) == 1.0


def test_ann_fidelity_truncates_to_k() -> None:
    # Three overlapping items, but only k=1 is compared.
    assert ann_recall_vs_exact(["a", "b", "c"], ["a", "x", "y"], k=1) == 1.0


# --------------------------------------------------------------------------- #
# mean
# --------------------------------------------------------------------------- #
def test_mean_of_values() -> None:
    assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_mean_empty_is_zero() -> None:
    assert mean([]) == 0.0


def test_mean_single_value() -> None:
    assert mean([0.42]) == pytest.approx(0.42)
