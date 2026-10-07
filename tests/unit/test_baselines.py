"""Unit tests for the evaluation baselines.

Baselines are only useful if they are deterministic: a random baseline
that changes between runs would make the evaluation report
un-reproducible and the CI gate unreliable. These tests pin
determinism, the tie-break rule, and the shape of the results.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from recsys.evaluation.baselines import (
    BaselineResult,
    PopularityProvider,
    SyntheticPopularityProvider,
    popularity_baseline,
    random_baseline,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _items(n: int) -> list[str]:
    return [f"i_{i:04d}" for i in range(n)]


# --------------------------------------------------------------------------- #
# random_baseline
# --------------------------------------------------------------------------- #
def test_random_baseline_is_deterministic() -> None:
    ids = _items(50)
    a = random_baseline(ids, k=10, query_id="q1")
    b = random_baseline(ids, k=10, query_id="q1")
    assert a == b


def test_random_baseline_differs_by_query_id() -> None:
    ids = _items(50)
    a = random_baseline(ids, k=10, query_id="q1")
    b = random_baseline(ids, k=10, query_id="q2")
    assert a.retrieved != b.retrieved


def test_random_baseline_ignores_input_order() -> None:
    ids = _items(20)
    a = random_baseline(ids, k=5, query_id="q")
    b = random_baseline(list(reversed(ids)), k=5, query_id="q")
    assert a.retrieved == b.retrieved


def test_random_baseline_respects_k() -> None:
    ids = _items(50)
    assert len(random_baseline(ids, k=3, query_id="q").retrieved) == 3
    assert len(random_baseline(ids, k=50, query_id="q").retrieved) == 50


def test_random_baseline_k_larger_than_catalog() -> None:
    ids = _items(5)
    r = random_baseline(ids, k=100, query_id="q")
    assert len(r.retrieved) == 5
    assert set(r.retrieved) == set(ids)


def test_random_baseline_empty_catalog() -> None:
    r = random_baseline([], k=10, query_id="q")
    assert r.retrieved == ()
    assert r.scores == ()


def test_random_baseline_non_positive_k() -> None:
    r = random_baseline(_items(10), k=0, query_id="q")
    assert r.retrieved == ()
    assert r.scores == ()


def test_random_baseline_returns_all_distinct_items() -> None:
    ids = _items(20)
    r = random_baseline(ids, k=20, query_id="q")
    assert len(set(r.retrieved)) == 20


def test_random_baseline_result_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    r = random_baseline(_items(5), k=2, query_id="q")
    with pytest.raises(FrozenInstanceError):
        r.retrieved = ()  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# synthetic popularity provider
# --------------------------------------------------------------------------- #
def test_synthetic_provider_deterministic() -> None:
    p = SyntheticPopularityProvider()
    a = p.scores_for(_items(20))
    b = p.scores_for(_items(20))
    assert a == b


def test_synthetic_provider_scores_in_expected_range() -> None:
    p = SyntheticPopularityProvider()
    scores = p.scores_for(_items(100))
    for s in scores.values():
        assert 0.0 <= s < 1000.0


def test_synthetic_provider_score_is_function_of_item_id() -> None:
    # The same item id gets the same score regardless of what else is
    # in the batch.
    p = SyntheticPopularityProvider()
    a = p.scores_for(["i_0001"])
    b = p.scores_for(["i_0001", "i_0002", "i_0003"])
    assert a["i_0001"] == b["i_0001"]


def test_synthetic_provider_satisfies_protocol() -> None:
    assert isinstance(SyntheticPopularityProvider(), PopularityProvider)


# --------------------------------------------------------------------------- #
# popularity_baseline
# --------------------------------------------------------------------------- #
def test_popularity_baseline_is_deterministic() -> None:
    ids = _items(50)
    a = popularity_baseline(ids, k=10)
    b = popularity_baseline(ids, k=10)
    assert a == b


def test_popularity_baseline_respects_k() -> None:
    ids = _items(50)
    assert len(popularity_baseline(ids, k=3).retrieved) == 3
    assert len(popularity_baseline(ids, k=50).retrieved) == 50


def test_popularity_baseline_orders_by_score_descending() -> None:
    ids = _items(20)
    r = popularity_baseline(ids, k=20)
    scores = list(r.scores)
    assert scores == sorted(scores, reverse=True)


def test_popularity_baseline_ties_broken_by_item_id() -> None:
    # Custom provider: all items score 42.0, so the tie-break is what
    # decides the order.
    class _FlatProvider:
        name = "flat"

        def scores_for(self, item_ids: Sequence[str]) -> dict[str, float]:
            return dict.fromkeys(item_ids, 42.0)

    ids = ["i_0003", "i_0001", "i_0002"]
    r = popularity_baseline(ids, k=3, provider=_FlatProvider())
    assert r.retrieved == ("i_0001", "i_0002", "i_0003")


def test_popularity_baseline_uses_default_provider() -> None:
    # Passing no provider uses the synthetic one.
    a = popularity_baseline(_items(10), k=5)
    b = popularity_baseline(_items(10), k=5, provider=SyntheticPopularityProvider())
    assert a == b


def test_popularity_baseline_missing_scores_treated_as_zero() -> None:
    # A provider that returns an empty mapping: every id is score 0.0,
    # the ranking is by item_id ascending.
    class _EmptyProvider:
        name = "empty"

        def scores_for(self, item_ids: Sequence[str]) -> dict[str, float]:
            _ = item_ids
            return {}

    ids = ["i_0003", "i_0001", "i_0002"]
    r = popularity_baseline(ids, k=3, provider=_EmptyProvider())
    assert r.retrieved == ("i_0001", "i_0002", "i_0003")
    assert r.scores == (0.0, 0.0, 0.0)


def test_popularity_baseline_empty_catalog() -> None:
    r = popularity_baseline([], k=5)
    assert r.retrieved == ()
    assert r.scores == ()


def test_popularity_baseline_non_positive_k() -> None:
    r = popularity_baseline(_items(10), k=0)
    assert r.retrieved == ()
    assert r.scores == ()


def test_popularity_baseline_k_larger_than_catalog() -> None:
    ids = _items(5)
    r = popularity_baseline(ids, k=100)
    assert len(r.retrieved) == 5
    assert set(r.retrieved) == set(ids)


# --------------------------------------------------------------------------- #
# BaselineResult
# --------------------------------------------------------------------------- #
def test_baseline_result_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    r = BaselineResult(retrieved=("a",), scores=(1.0,))
    with pytest.raises(FrozenInstanceError):
        r.retrieved = ()  # type: ignore[misc]


def test_baseline_result_equal_when_fields_equal() -> None:
    a = BaselineResult(retrieved=("a", "b"), scores=(2.0, 1.0))
    b = BaselineResult(retrieved=("a", "b"), scores=(2.0, 1.0))
    assert a == b
