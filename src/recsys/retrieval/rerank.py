"""Re-ranking interface.

M2 does not implement re-ranking. ADR-0005 records the decision to defer it
to M3, where it ships as an experiment arm alongside the serving API. This
module exists so that M3 has a defined place to plug in without a call-site
migration across the retrieval layer.

It is deliberately empty of implementations. The retrieval layer does not
import it in M2.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """A strategy that reorders a candidate list.

    Implementations will be provided in M3. The protocol is defined now so
    that the interface is designed under low pressure, rather than under
    the pressure of a shipped serving path.

    The contract:

    - Input: a query context (the seed items, the user id, any request
      metadata a re-ranker may consult) and a candidate list of
      ``(item_id, score)`` pairs already sorted by score descending.
    - Output: the same candidates in a new order, still as ``(item_id,
      score)`` pairs. A re-ranker must not add or remove items; it only
      reorders. Filtering is retrieval's job; diversity re-ranking that
      drops items is a future extension and will need a different protocol.
    - Purity: given the same inputs, an implementation must produce the
      same output. Determinism is required so that the M5 A/B assignment
      is meaningful.
    """

    name: str

    def rerank(
        self,
        *,
        seed_item_ids: list[str],
        candidates: list[tuple[str, float]],
        k: int,
    ) -> list[tuple[str, float]]:
        """Return ``candidates`` reordered, truncated to at most ``k`` items."""
        ...
