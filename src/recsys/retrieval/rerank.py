"""Re-ranking interface and configuration (ADR-0016).

M2 defined an empty ``Reranker`` protocol so that M3 had a defined
plug-in point. This module fills it: a ``Candidate`` value object
that carries every signal the re-ranker consults, a frozen
``RerankConfig`` that validates itself, and the protocol every
concrete re-ranker implements.

The re-ranker is a **pure function of its inputs**. Given the same
``Candidate`` list and the same ``RerankConfig``, an implementation
produces the same output list byte-for-byte, including tie order
(ADR-0016 § Determinism). Every sort is ``(score DESC, item_id
ASC)``: a score tie is resolved by the item id, ascending,
byte-for-byte. This is what makes the M2 evaluation gate, the M5
A/B assignment, and post-deploy diagnosis meaningful.

Normalization helpers live here, not inside an implementation,
because both the concrete re-ranker (M3.4.4) and the tests need the
exact same arithmetic. A helper that is copied into a second place
is a helper that drifts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

#: The value added to a min-max divisor to keep it non-zero when the
#: window is degenerate (``max == min``). 1e-8 is small enough that it
#: does not measurably change a normal window and large enough to keep
#: the divisor representable in float32 and float64.
MINMAX_EPS: float = 1e-8

#: The neutral normalization value for a signal that carries no
#: information: a degenerate window, or a provider that raised or
#: timed out. Every candidate gets this value, so the signal's weight
#: contributes the same amount to every blended score and does not
#: change the ordering (ADR-0016 § Step 1, § Failure behavior).
NEUTRAL_NORM: float = 0.5


@dataclass(frozen=True)
class Candidate:
    """One candidate and every signal the re-ranker consults.

    All three scalar signals are "larger is better" after
    normalization; the raw values below carry the natural polarity of
    their source.

    ``similarity`` is a **cosine similarity in ``[-1, 1]``**, higher
    is better. Conversion from a distance (the ``PgvectorBackend``
    returns ``vector_cosine_ops`` distances, ``distance = 1 -
    similarity``) happens **in the caller**, before this object is
    constructed. The re-ranker never guesses at the metric; it reads
    the field and treats it as similarity.

    ``popularity`` is the raw provider value (M3: a synthetic score;
    M5: an event count). The re-ranker normalizes it by rank within
    the window, so the raw scale does not matter — only the ordering
    does.

    ``age_days`` is measured from ``item.created_at``. An edit that
    moves ``updated_at`` is not "the item is new"; a future ADR may
    add a second decay term for metadata freshness, but the signal
    here is catalog introduction.

    ``vector`` is optional. It is L2-normalized ``float32`` of the
    same dimension the index uses, and it is only read when MMR
    runs. A backend that does not expose candidate vectors leaves it
    as ``None``; MMR is skipped and a metric records the skip.

    **The dataclass is frozen, but the array inside is not.** A
    frozen dataclass prevents reassigning ``candidate.vector``; it
    does not make the array immutable. Nothing in the re-ranker
    mutates it, and the retrieval layer must not hand out an array
    it continues to write to. The same caveat applies to a caller
    that stores the same array in two candidates.
    """

    item_id: str
    similarity: float
    popularity: float
    age_days: float
    vector: NDArray[np.float32] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must be a non-empty string")
        if not math.isfinite(self.similarity):
            raise ValueError(f"similarity must be finite, got {self.similarity!r}")
        if not math.isfinite(self.popularity):
            raise ValueError(f"popularity must be finite, got {self.popularity!r}")
        if not math.isfinite(self.age_days):
            raise ValueError(f"age_days must be finite, got {self.age_days!r}")
        if self.age_days < 0:
            # A negative age is a data bug (created_at in the future);
            # clamping would hide it. Fail loudly.
            raise ValueError(f"age_days must be >= 0, got {self.age_days!r}")


@dataclass(frozen=True)
class RerankConfig:
    """The re-ranker's configuration (ADR-0016 § Config validation).

    Every field is validated in ``__post_init__``; a malformed
    config is a startup failure, not a warning. A re-ranker with a
    negative weight produces nonsense scores that are worse than not
    re-ranking at all, and silently falling back to "no re-ranking"
    would hide the misconfiguration behind a metric that only moves
    by a few points.

    The values that produced a response are echoed in
    ``meta.rerank`` so a client can tell which configuration it saw.
    """

    w_sim: float = 0.7
    w_pop: float = 0.2
    w_rec: float = 0.1
    mmr_lambda: float = 0.7
    mmr_window: int = 50
    mmr_min_k: int = 10
    candidate_multiplier: int = 4
    recency_half_life_days: int = 90
    enable_mmr: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("w_sim", self.w_sim),
            ("w_pop", self.w_pop),
            ("w_rec", self.w_rec),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")

        weight_sum = self.w_sim + self.w_pop + self.w_rec
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"w_sim + w_pop + w_rec must equal 1.0 within 1e-6, got {weight_sum!r}"
            )

        if not math.isfinite(self.mmr_lambda):
            raise ValueError(f"mmr_lambda must be finite, got {self.mmr_lambda!r}")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be in [0.0, 1.0], got {self.mmr_lambda!r}")

        if self.recency_half_life_days <= 0:
            raise ValueError(
                f"recency_half_life_days must be > 0, got {self.recency_half_life_days!r}"
            )
        if self.candidate_multiplier < 1:
            raise ValueError(
                f"candidate_multiplier must be >= 1, got {self.candidate_multiplier!r}"
            )
        if self.mmr_window < 1:
            raise ValueError(f"mmr_window must be >= 1, got {self.mmr_window!r}")
        if self.mmr_min_k < 1:
            raise ValueError(f"mmr_min_k must be >= 1, got {self.mmr_min_k!r}")


@runtime_checkable
class Reranker(Protocol):
    """A strategy that reorders a candidate list (ADR-0016).

    Contract:

    - **Input:** a ``Candidate`` list (every signal the re-ranker
      consults is already on the object) and ``k``, the number of
      items the caller wants back.
    - **Output:** at most ``k`` ``(item_id, score)`` pairs, sorted by
      ``(score DESC, item_id ASC)``. The score is the blended (or
      MMR) score; the pair's second element is what the caller shows
      as the item's ranking score. A re-ranker returns a subset of
      the input ids, reordered: it does not add or remove items
      beyond truncation, and it does not invent ids.
    - **Purity:** given the same input, the same output, including
      tie order. The re-ranker reads no clock, opens no connection,
      and consults no process-wide state. Providers supply the
      time-dependent values (``age_days``) before the candidate is
      constructed.
    """

    name: str

    def rerank(
        self,
        *,
        candidates: list[Candidate],
        k: int,
    ) -> list[tuple[str, float]]:
        """Return at most ``k`` ``(item_id, score)`` pairs, sorted."""
        ...


# ------------------------------------------------------------------ #
# normalization helpers (pure)
# ------------------------------------------------------------------ #
def minmax_norm(values: list[float]) -> list[float]:
    """Return min-max normalized ``values`` in ``[0, 1]``.

    Degenerate windows (``max == min``, including a single-value
    list) return ``NEUTRAL_NORM`` for every entry. The choice is the
    neutral one: the signal contributes the same amount to every
    blended score and therefore does not change the ordering.
    Returning ``0.0`` would silently remove the signal's weight;
    returning ``1.0`` would give it full weight for no information.

    The epsilon on the divisor is the second line of defense: the
    degenerate case is caught first, but a window where ``max`` and
    ``min`` differ by less than float precision still divides by a
    non-zero number.
    """
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [NEUTRAL_NORM] * len(values)
    span = hi - lo + MINMAX_EPS
    return [(v - lo) / span for v in values]


def rank_norm(values: list[float]) -> list[float]:
    """Return rank-based normalized ``values`` in ``[0, 1]``, higher-is-better.

    The largest value gets ``1.0``, the smallest gets ``0.0``, and
    ties share the *average* of the ranks they span. Averaging is
    what makes the result deterministic when two candidates share a
    value: without it, the assignment would depend on input order.

    A single-value list is degenerate and returns ``NEUTRAL_NORM``.
    The helper is used for ``popularity``, whose distribution is
    Zipf-shaped and where a min-max would be dominated by a single
    outlier (ADR-0016 § Step 1).
    """
    if not values:
        return []
    n = len(values)
    if n == 1:
        return [NEUTRAL_NORM]

    # Sort indices by value descending; assign average rank to ties.
    order = sorted(range(n), key=lambda i: (-values[i], i))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    # rank 0 (largest) -> 1.0; rank n-1 (smallest) -> 0.0
    return [1.0 - (r / (n - 1)) for r in ranks]


def recency_decay(age_days: float, *, half_life_days: int) -> float:
    """Return ``0.5 ** (age_days / half_life_days)``.

    An item added today has ``1.0``; an item added one half-life ago
    has ``0.5``. The exponential form has one parameter and does not
    reach zero, so an old item is deprioritized but never excluded.

    ``half_life_days`` must be positive; ``RerankConfig`` enforces
    that at construction, and the check here is a second guard for a
    caller that constructs a config by hand.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be > 0, got {half_life_days!r}")
    if age_days < 0:
        raise ValueError(f"age_days must be >= 0, got {age_days!r}")
    # float ** float returns Any in typeshed (int ** int can be
    # complex); the float() call pins the actual runtime type.
    return float(0.5 ** (age_days / half_life_days))


__all__ = [
    "MINMAX_EPS",
    "NEUTRAL_NORM",
    "Candidate",
    "RerankConfig",
    "Reranker",
    "minmax_norm",
    "rank_norm",
    "recency_decay",
]
