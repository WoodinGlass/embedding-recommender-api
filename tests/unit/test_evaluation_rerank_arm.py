"""Unit tests for :func:`evaluate_rerank_arm` (ADR-0016, ADR-0010).

The runner is exercised with a fake backend and frozen providers so
the tests run in Colab without a database, and so the assertions are
about the runner's behavior, not about a model's quality.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.evaluation.golden_set import GoldenQuery, GoldenSet
from recsys.evaluation.runner import (
    RerankArmResult,
    evaluate_rerank_arm,
)
from recsys.retrieval.providers import (
    FrozenPopularityProvider,
    FrozenRecencyProvider,
    ProviderError,
)
from recsys.retrieval.rerank import (
    RerankConfig,
    WeightedBlendReranker,
)


# ---------------------------------------------------------------- #
# fixtures
# ---------------------------------------------------------------- #
def _vec(x: float, y: float) -> NDArray[np.float32]:
    v = np.array([x, y], dtype=np.float32)
    return v / float(np.linalg.norm(v))


class FakeBackend:
    """Returns a fixed ranking per query, ignoring the query vector.

    The runner is not testing the search; it is testing what the
    runner does with whatever search returns. A fixed ranking makes
    the assertions deterministic and independent of the embedding.
    """

    name = "fake"

    def __init__(self, ranked: list[str]) -> None:
        self._ranked = ranked
        self.calls = 0

    def is_ready(self) -> bool:
        return True

    def search(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        self.calls += 1
        return [(iid, 1.0 - i * 0.01) for i, iid in enumerate(self._ranked[:k])]


class FakeBackendWithVectors(FakeBackend):
    def search_with_vectors(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float, NDArray[np.float32]]]:
        self.calls += 1
        out: list[tuple[str, float, NDArray[np.float32]]] = []
        for i, iid in enumerate(self._ranked[:k]):
            vec = _vec(1.0, 0.001 * i)
            out.append((iid, 1.0 - i * 0.01, vec))
        return out


def _golden() -> GoldenSet:
    return GoldenSet(
        version="v1",
        path=pathlib.Path("v1.jsonl"),
        queries=(
            GoldenQuery(
                query_id="q1",
                topic="t",
                seed_item_ids=("i_0",),
                relevant_item_ids=("i_a", "i_b"),
            ),
        ),
    )


def _embedding_lookup(item_id: str) -> NDArray[np.float32] | None:
    # Every item resolves; the runner is not testing missing embeddings.
    return _vec(1.0, 0.0)


def _config(**overrides: Any) -> RerankConfig:
    base: dict[str, Any] = {
        "w_sim": 0.7,
        "w_pop": 0.2,
        "w_rec": 0.1,
        "mmr_lambda": 0.7,
        "mmr_window": 50,
        "mmr_min_k": 10,
        "candidate_multiplier": 2,
        "recency_half_life_days": 90,
        "enable_mmr": False,
    }
    base.update(overrides)
    return RerankConfig(**base)


def _pop(score: Mapping[str, float] | None = None) -> FrozenPopularityProvider:
    return FrozenPopularityProvider(score or {"i_a": 10.0, "i_b": 1.0})


def _rec(age: Mapping[str, float] | None = None) -> FrozenRecencyProvider:
    return FrozenRecencyProvider(age or {"i_a": 1.0, "i_b": 1.0})


def _run(**overrides: Any) -> RerankArmResult:
    defaults: dict[str, Any] = {
        "backend": FakeBackend(["i_a", "i_b", "i_0"]),
        "golden_set": _golden(),
        "embedding_lookup": _embedding_lookup,
        "k": 2,
        "config": _config(),
        "reranker": WeightedBlendReranker(_config()),
        "popularity_provider": _pop(),
        "recency_provider": _rec(),
    }
    defaults.update(overrides)
    return evaluate_rerank_arm(**defaults)


# ---------------------------------------------------------------- #
# happy path
# ---------------------------------------------------------------- #
def test_arm_runs_and_produces_aggregate() -> None:
    result = _run()
    assert result.status == "ok"
    assert result.error_message is None
    assert "recall_at_2" in result.aggregate
    assert "ndcg_at_2" in result.aggregate
    assert "mrr" in result.aggregate


def test_ann_recall_vs_exact_is_null_post_rerank() -> None:
    """The reranker deliberately changes the top-k; fidelity post-rerank
    measures the wrong thing (ADR-0016 § Step 3). The runner reports
    None for that field, never 0.0 and never a partial number."""
    result = _run()
    assert result.aggregate["ann_recall_vs_exact"] is None


def test_provider_missing_rates_are_zero_when_providers_answer() -> None:
    result = _run()
    assert result.aggregate["provider_missing_rate_popularity"] == 0.0
    assert result.aggregate["provider_missing_rate_recency"] == 0.0


# ---------------------------------------------------------------- #
# provider failure -> neutral, rate recorded, status degraded
# ---------------------------------------------------------------- #
def test_popularity_provider_failure_is_neutral_and_recorded() -> None:
    class BoomPop:
        name = "boom"

        def get(self, item_ids: Any) -> Mapping[str, float]:
            raise ProviderError("forced")

    result = _run(popularity_provider=BoomPop())
    assert result.status == "degraded"
    assert result.aggregate["provider_missing_rate_popularity"] == 1.0


def test_recency_provider_failure_is_neutral_and_recorded() -> None:
    class BoomRec:
        name = "boom"

        def get(self, item_ids: Any) -> Mapping[str, float]:
            raise ProviderError("forced")

    result = _run(recency_provider=BoomRec())
    assert result.status == "degraded"
    assert result.aggregate["provider_missing_rate_recency"] == 1.0


# ---------------------------------------------------------------- #
# pre-rerank fidelity
# ---------------------------------------------------------------- #
def test_pre_rerank_fidelity_reported_when_exact_backend_given() -> None:
    exact = FakeBackend(["i_a", "i_b", "i_0"])
    result = _run(exact_backend=exact)
    assert "ann_recall_vs_exact_pre_rerank" in result.aggregate
    assert result.aggregate["ann_recall_vs_exact_pre_rerank"] is not None


def test_pre_rerank_fidelity_absent_without_exact_backend() -> None:
    result = _run(exact_backend=None)
    assert "ann_recall_vs_exact_pre_rerank" not in result.aggregate


# ---------------------------------------------------------------- #
# MMR needs vectors
# ---------------------------------------------------------------- #
def test_mmr_arm_without_search_with_vectors_errors() -> None:
    """An arm that needs vectors on a backend that cannot supply them
    is an error, not a crash. The report carries status="error" so
    the failure is visible and the whole evaluation does not stop."""
    cfg = _config(enable_mmr=True, mmr_min_k=1)
    result = _run(
        backend=FakeBackend(["i_a", "i_b", "i_0"]),  # no search_with_vectors
        config=cfg,
        reranker=WeightedBlendReranker(cfg),
    )
    assert result.status == "error"
    assert result.error_message is not None
    assert "search_with_vectors" in result.error_message


def test_mmr_arm_with_vectors_runs() -> None:
    cfg = _config(enable_mmr=True, mmr_min_k=1)
    result = _run(
        backend=FakeBackendWithVectors(["i_a", "i_b", "i_0"]),
        config=cfg,
        reranker=WeightedBlendReranker(cfg),
    )
    assert result.status == "ok"


# ---------------------------------------------------------------- #
# guards
# ---------------------------------------------------------------- #
def test_zero_k_raises() -> None:
    from recsys.evaluation.runner import EvaluationError

    with pytest.raises(EvaluationError, match="k must be positive"):
        _run(k=0)
