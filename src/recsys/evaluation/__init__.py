"""Offline evaluation for retrieval.

The evaluation harness measures recall@k, NDCG@k, MRR, and ANN fidelity
against a versioned golden set. Definitions are in
``docs/adr/0009-golden-set-and-metrics.md``; the gate that consumes the
results is in ``docs/adr/0010-evaluation-thresholds.md``.
"""

from recsys.evaluation.baselines import (
    BaselineResult,
    PopularityProvider,
    SyntheticPopularityProvider,
    popularity_baseline,
    random_baseline,
)
from recsys.evaluation.metrics import (
    ann_recall_vs_exact,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

__all__ = [
    "BaselineResult",
    "PopularityProvider",
    "SyntheticPopularityProvider",
    "ann_recall_vs_exact",
    "mrr",
    "ndcg_at_k",
    "popularity_baseline",
    "random_baseline",
    "recall_at_k",
]
