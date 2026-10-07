"""Baselines for the evaluation table.

ADR-0009 § 4 fixes three baselines:

- **Random** — deterministic per query id. The floor: a system that does
  not beat random is worse than noise.
- **Synthetic popularity** — deterministic per item id. A placeholder for
  a real event-based popularity signal, which lands in M5. The interface
  (:class:`PopularityProvider`) is the seam that M5 will swap.
- **Exact kNN** — brute-force cosine over the same vectors. Not a
  function in this module: it is :class:`recsys.retrieval.numpy_backend.NumpyBackend`
  called with the same query vector the production path would use. The
  evaluation runner calls it directly. It is listed here in comments so
  the three baselines are documented together.

The module is pure: no I/O, no configuration, no state. Determinism is
the whole point — a baseline that changes between runs would make the
report un-reproducible and the CI gate unreliable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

#: The modulus used by :class:`SyntheticPopularityProvider`. Chosen so
#: scores span ``[0, 1000)`` and the distribution is flat; a real
#: popularity distribution is heavily skewed and the M5 provider will
#: look nothing like this.
_SYNTHETIC_POPULARITY_MODULUS = 1000


@dataclass(frozen=True)
class BaselineResult:
    """The output of a baseline for a single query.

    ``retrieved`` is the ranked list of item ids, ``scores`` is the
    parallel list of scores used for ranking. Scores are not comparable
    across baselines (random uses no score, popularity uses a pseudo-score,
    exact kNN uses cosine similarity); they are kept so a caller can log
    them or use them for tie-breaking diagnostics.
    """

    retrieved: tuple[str, ...]
    scores: tuple[float, ...]


# --------------------------------------------------------------------------- #
# popularity
# --------------------------------------------------------------------------- #
@runtime_checkable
class PopularityProvider(Protocol):
    """Source of item popularity scores.

    M2 ships one implementation, :class:`SyntheticPopularityProvider`,
    whose scores are a deterministic function of the item id. M5 will add
    an event-based provider; the seam exists so that swapping it does not
    touch the evaluation runner or the metrics.
    """

    name: str

    def scores_for(self, item_ids: Sequence[str]) -> dict[str, float]:
        """Return a ``{item_id: score}`` mapping for the given ids.

        Scores are compared with ``>``; higher is more popular. Missing
        ids are treated as score ``0.0`` by callers, so an implementation
        may return a subset without changing the ranking of the rest.
        """
        ...


class SyntheticPopularityProvider:
    """Deterministic pseudo-popularity.

    The score for an item is ``int(sha256(item_id)[:8], 16) % 1000``.
    This is **not** real popularity: real popularity is concentrated and
    the synthetic score is flat. It is a placeholder with the right shape
    (a number per item, higher is better) so the evaluation harness can
    run before M5, and so the README table has a mid-baseline between
    random and the embedding model.
    """

    name: str = "synthetic"

    def scores_for(self, item_ids: Sequence[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for iid in item_ids:
            digest = hashlib.sha256(iid.encode("utf-8")).hexdigest()
            out[iid] = float(int(digest[:8], 16) % _SYNTHETIC_POPULARITY_MODULUS)
        return out


# --------------------------------------------------------------------------- #
# random
# --------------------------------------------------------------------------- #
def _random_rank(item_ids: Sequence[str], query_id: str) -> list[str]:
    """Return ``item_ids`` in a deterministic order derived from ``query_id``.

    Two properties matter:

    - The same ``(item_ids, query_id)`` pair produces the same order on
      any machine, in any process. This is what makes the random baseline
      reproducible.
    - The order is a function of the *set* of item ids, not of their input
      order, so that a caller who sorts differently does not change the
      baseline. The sort below is a canonicalization, not a ranking.

    The implementation hashes ``query_id`` and each ``item_id`` with a
    shared prefix so the outcome depends on both. Using ``sorted(...)``
    on the hash keeps the result independent of the input order.
    """
    prefix = query_id.encode("utf-8")
    return sorted(
        item_ids,
        key=lambda iid: hashlib.sha256(prefix + b"\x00" + iid.encode("utf-8")).digest(),
    )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def random_baseline(
    item_ids: Sequence[str],
    *,
    k: int,
    query_id: str,
) -> BaselineResult:
    """Return the top ``k`` of a deterministic pseudo-random ordering.

    ``item_ids`` is the full catalog. The result depends only on the
    sorted set of ids and on ``query_id``; the input order is irrelevant.
    """
    if k <= 0:
        return BaselineResult(retrieved=(), scores=())
    if not item_ids:
        return BaselineResult(retrieved=(), scores=())
    ranked = _random_rank(item_ids, query_id)[:k]
    # No meaningful scores in a random ranking; use the position as a
    # placeholder so a caller that wants a parallel array has one.
    scores = tuple(float(k - i) for i in range(len(ranked)))
    return BaselineResult(retrieved=tuple(ranked), scores=scores)


def popularity_baseline(
    item_ids: Sequence[str],
    *,
    k: int,
    provider: PopularityProvider | None = None,
) -> BaselineResult:
    """Return the top ``k`` items by popularity score.

    Ties are broken by ``item_id`` ascending, so the result is
    deterministic for any provider that returns the same scores.
    """
    if k <= 0 or not item_ids:
        return BaselineResult(retrieved=(), scores=())
    if provider is None:
        provider = SyntheticPopularityProvider()

    scores = provider.scores_for(item_ids)
    # Missing ids are score 0.0, matching the protocol's contract.
    ordered = sorted(
        item_ids,
        key=lambda iid: (-scores.get(iid, 0.0), iid),
    )
    ranked = ordered[:k]
    return BaselineResult(
        retrieved=tuple(ranked),
        scores=tuple(scores.get(iid, 0.0) for iid in ranked),
    )


__all__ = [
    "BaselineResult",
    "PopularityProvider",
    "SyntheticPopularityProvider",
    "popularity_baseline",
    "random_baseline",
]
