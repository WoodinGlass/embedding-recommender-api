"""Unit tests for the weighted blend re-ranker (ADR-0016).

MMR tests land in M3.4.4b once the step is implemented; the config
that requests it is rejected at construction here.
"""

from __future__ import annotations

import pytest

from recsys.retrieval.rerank import (
    NEUTRAL_NORM,
    Candidate,
    RerankConfig,
    RerankError,
    WeightedBlendReranker,
    minmax_norm,
    rank_norm,
    recency_decay,
)


def _c(
    item_id: str,
    *,
    similarity: float = 0.5,
    popularity: float = 0.0,
    age_days: float = 0.0,
) -> Candidate:
    return Candidate(
        item_id=item_id,
        similarity=similarity,
        popularity=popularity,
        age_days=age_days,
    )


def _cfg(**overrides: object) -> RerankConfig:
    base: dict[str, object] = {
        "w_sim": 0.7,
        "w_pop": 0.2,
        "w_rec": 0.1,
        "mmr_lambda": 0.7,
        "mmr_window": 50,
        "mmr_min_k": 10,
        "candidate_multiplier": 4,
        "recency_half_life_days": 90,
        "enable_mmr": False,
    }
    base.update(overrides)
    return RerankConfig(**base)  # type: ignore[arg-type]


def _reranker(cfg: RerankConfig | None = None) -> WeightedBlendReranker:
    return WeightedBlendReranker(cfg or _cfg())


# ---------------------------------------------------------------- #
# config validation
# ---------------------------------------------------------------- #
def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"w_sim \+ w_pop \+ w_rec"):
        _cfg(w_sim=0.5, w_pop=0.5, w_rec=0.5)


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="w_sim"):
        _cfg(w_sim=-0.1, w_pop=0.6, w_rec=0.5)


def test_mmr_lambda_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="mmr_lambda"):
        _cfg(mmr_lambda=1.5)


def test_enable_mmr_true_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="MMR"):
        WeightedBlendReranker(_cfg(enable_mmr=True))


# ---------------------------------------------------------------- #
# guards
# ---------------------------------------------------------------- #
def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        _reranker().rerank(candidates=[_c("i_1")], k=0)


def test_empty_candidates_returns_empty_list() -> None:
    assert _reranker().rerank(candidates=[], k=10) == []


# ---------------------------------------------------------------- #
# blend ordering
# ---------------------------------------------------------------- #
def test_similarity_drives_ordering_when_other_signals_equal() -> None:
    cands = [
        _c("i_low", similarity=0.1, popularity=5.0, age_days=0.0),
        _c("i_high", similarity=0.9, popularity=5.0, age_days=0.0),
        _c("i_mid", similarity=0.5, popularity=5.0, age_days=0.0),
    ]
    out = _reranker().rerank(candidates=cands, k=3)
    assert [item for item, _ in out] == ["i_high", "i_mid", "i_low"]


def test_rank_based_popularity_promotes_higher_popularity() -> None:
    cands = [
        _c("i_pop", similarity=0.5, popularity=1000.0),
        _c("i_mid", similarity=0.5, popularity=100.0),
        _c("i_low", similarity=0.5, popularity=1.0),
    ]
    out = _reranker().rerank(candidates=cands, k=3)
    assert [item for item, _ in out] == ["i_pop", "i_mid", "i_low"]


def test_recency_promotes_newer_items() -> None:
    cands = [
        _c("i_new", similarity=0.5, popularity=5.0, age_days=0.0),
        _c("i_old", similarity=0.5, popularity=5.0, age_days=180.0),
    ]
    out = _reranker().rerank(candidates=cands, k=2)
    assert [item for item, _ in out] == ["i_new", "i_old"]


def test_missing_popularity_signal_is_neutral() -> None:
    cands = [
        _c("i_hi_sim", similarity=0.9, popularity=1.0),
        _c("i_lo_sim", similarity=0.1, popularity=1_000_000.0),
    ]
    out = _reranker().rerank(candidates=cands, k=2, missing_signals=frozenset({"popularity"}))
    assert [item for item, _ in out] == ["i_hi_sim", "i_lo_sim"]


def test_missing_recency_signal_is_neutral() -> None:
    cands = [
        _c("i_new_low_sim", similarity=0.1, popularity=5.0, age_days=0.0),
        _c("i_old_hi_sim", similarity=0.9, popularity=5.0, age_days=1000.0),
    ]
    out = _reranker().rerank(candidates=cands, k=2, missing_signals=frozenset({"recency"}))
    assert [item for item, _ in out] == ["i_old_hi_sim", "i_new_low_sim"]


def test_unknown_missing_signal_ignored() -> None:
    cands = [
        _c("i_hi", similarity=0.9, popularity=5.0),
        _c("i_lo", similarity=0.1, popularity=5.0),
    ]
    out = _reranker().rerank(candidates=cands, k=2, missing_signals=frozenset({"unknown_signal"}))
    assert [item for item, _ in out] == ["i_hi", "i_lo"]


# ---------------------------------------------------------------- #
# truncation and tie-break
# ---------------------------------------------------------------- #
def test_truncates_to_k() -> None:
    cands = [_c(f"i_{i}", similarity=0.5) for i in range(10)]
    out = _reranker().rerank(candidates=cands, k=3)
    assert len(out) == 3


def test_k_larger_than_candidates_returns_all() -> None:
    cands = [_c("i_1"), _c("i_2")]
    out = _reranker().rerank(candidates=cands, k=100)
    assert len(out) == 2


def test_tie_break_by_item_id_ascending() -> None:
    """Two candidates with identical signals share a blend score; the
    total order resolves them by item_id ascending."""
    cands = [
        _c("i_c", similarity=0.5, popularity=5.0, age_days=1.0),
        _c("i_a", similarity=0.5, popularity=5.0, age_days=1.0),
        _c("i_b", similarity=0.5, popularity=5.0, age_days=1.0),
    ]
    out = _reranker().rerank(candidates=cands, k=3)
    assert [item for item, _ in out] == ["i_a", "i_b", "i_c"]


# ---------------------------------------------------------------- #
# helper coverage
# ---------------------------------------------------------------- #
def test_minmax_degenerate_returns_neutral() -> None:
    assert minmax_norm([3.0, 3.0, 3.0]) == [NEUTRAL_NORM] * 3


def test_rank_norm_averages_ties() -> None:
    out = rank_norm([10.0, 10.0, 1.0])
    assert out[0] == out[1] == 0.75
    assert out[2] == 0.0


def test_recency_decay_half_life() -> None:
    assert recency_decay(0.0, half_life_days=90) == 1.0
    assert recency_decay(90.0, half_life_days=90) == 0.5
    assert recency_decay(180.0, half_life_days=90) == 0.25


def test_rerank_error_carries_step() -> None:
    err = RerankError(step="blend", cause=RuntimeError("x"))
    assert err.step == "blend"
    assert isinstance(err.cause, RuntimeError)
