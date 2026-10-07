"""Retrieval metrics.

Pure functions from a ranked list of retrieved item ids and a set of
relevant item ids to a scalar in ``[0, 1]``. No I/O, no configuration, no
state. Definitions and conventions are fixed by
``docs/adr/0009-golden-set-and-metrics.md`` § 2.

Conventions worth restating here because they are easy to get wrong:

- **Recall@k denominator is ``min(|relevant|, k)``.** If a query has more
  relevant items than ``k``, a perfect top-k cannot recall all of them;
  the denominator reflects that.
- **NDCG uses binary relevance.** An item is relevant or it is not; there
  are no graded judgments. The ideal DCG is the DCG of all relevant items
  placed first, truncated to ``k``.
- **MRR is over the full retrieved list, not truncated to k.** In this
  project the retrieval layer never returns more than ``k`` results, so
  the distinction does not arise in practice; the function is defined over
  whatever list it is given.
- **Ties in the retrieved list are the caller's responsibility.** These
  functions respect the order they are given. The retrieval layer sorts
  by score descending with a deterministic tie-break (item_id ascending,
  see ADR-0012).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(
    retrieved: Sequence[str],
    relevant: set[str] | frozenset[str],
    k: int,
) -> float:
    """Fraction of relevant items found in the top ``k``.

    ``min(|relevant|, k)`` is the denominator. Returns ``1.0`` when
    ``relevant`` is empty (a query with no ground truth is trivially
    recalled), and ``0.0`` for non-positive ``k``.
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 1.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: set[str] | frozenset[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at ``k`` with binary relevance.

    Returns ``0.0`` for non-positive ``k`` or an empty ``relevant`` set.
    """
    if k <= 0 or not relevant:
        return 0.0

    dcg = 0.0
    for i, item_id in enumerate(retrieved[:k]):
        if item_id in relevant:
            # rank i is 0-based; log2(rank + 2) is the standard denominator
            # for a 1-based rank: log2(1 + 1) = 1 for rank 1.
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: all relevant items first, truncated at k.
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(
    retrieved: Sequence[str],
    relevant: set[str] | frozenset[str],
) -> float:
    """Reciprocal rank of the first relevant item, or ``0.0`` if none.

    The retrieval layer in this project returns at most ``k`` results, so
    this function never sees a longer list than the caller's top-k. The
    definition is written for a general list anyway; it is cheaper to
    reason about than one that assumes a length.
    """
    if not relevant:
        return 0.0
    for i, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            return 1.0 / i
    return 0.0


def ann_recall_vs_exact(
    approx: Sequence[str],
    exact: Sequence[str],
    k: int,
) -> float:
    """Overlap of the top-``k`` of two rankings, as a fraction of ``k``.

    This is the ANN-fidelity metric from ADR-0009: it compares an
    approximate index to exact search over the *same* vectors. It is
    orthogonal to retrieval quality — a system with poor vectors can
    still have perfect ANN fidelity.

    Both inputs are truncated to ``k``. Returns ``1.0`` when both are
    empty (nothing to disagree on), and ``0.0`` for non-positive ``k``.
    """
    if k <= 0:
        return 0.0
    a = set(approx[:k])
    b = set(exact[:k])
    if not a and not b:
        return 1.0
    # Intersection over k, not over the union: an approximate top-k with
    # only 3 items when exact had 10 should not be rewarded for "recall
    # of what it returned".
    return len(a & b) / k


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, or ``0.0`` for an empty sequence.

    A separate function so that the runner never divides by zero and so
    the "empty input means zero" convention is stated once.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)
