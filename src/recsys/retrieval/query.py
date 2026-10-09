"""Query vector resolution (ADR-0009 § 3, ADR-0012).

The query vector for a recommend request is the mean of its seed
items' embeddings, L2-normalized. ADR-0009 fixes the rule: mean
over the seeds, then normalize, so the vector is on the unit sphere
the index was built for.

A seed whose item row is missing from the catalog is skipped; the
rule is the same one ``evaluation.runner.encode_query`` uses for a
missing seed embedding (a partially available seed set still
produces a usable query). If every seed is missing, the function
returns ``None`` and the caller decides what that means: the handler
treats it as a cold-start user and moves to the fallback chain
(ADR-0020 § Tier 3 and § Tier 4).

The function is synchronous; the handler calls it inside the bounded
thread pool it already uses for the rest of the synchronous
pipeline. See ADR-0012 (amended in M3.6.6f) for why the whole
pipeline shares one thread hop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

#: The separator between title and description when a seed item is
#: turned into the text the encoder reads. Must match what
#: ``recsys.embeddings.preprocess`` does for the catalog, so a seed's
#: embedding in the index and its query-side embedding are produced
#: from the same string. A mismatch would silently degrade retrieval
#: without failing; the constant is the single place the rule lives.
SEED_TEXT_SEPARATOR: str = " | "


class _EncoderLike(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


def resolve_query_vector(
    connection: Any,
    *,
    encoder: _EncoderLike,
    seed_item_ids: Sequence[str],
) -> NDArray[np.float32] | None:
    """Return the L2-normalized mean embedding of the seed items.

    ``connection`` is a synchronous psycopg connection. The query is
    bounded by ``FILTER_FIELDS``-style parameterization (a single
    ``ANY(%s)`` on the primary key), not by user-supplied text.

    Returns ``None`` when:
    - ``seed_item_ids`` is empty (a cold-start user with no seeds), or
    - every seed is missing from the catalog.

    Returns a ``float32`` array of the encoder's dimension otherwise.
    """
    if not seed_item_ids:
        return None

    with connection.cursor() as cur:
        cur.execute(
            "SELECT item_id, title, description FROM item WHERE item_id = ANY(%s)",
            (list(seed_item_ids),),
        )
        rows = cur.fetchall()

    if not rows:
        return None

    # Sort by item_id so the order is stable regardless of the
    # query plan; the mean is order-independent, but a stable order
    # makes logs and any future debug output reproducible.
    sorted_rows = sorted(rows, key=lambda r: str(r[0]))
    texts = [
        f"{str(row[1]).strip()}{SEED_TEXT_SEPARATOR}{str(row[2]).strip()}" for row in sorted_rows
    ]

    stacked: NDArray[np.float32] = encoder.encode(texts)
    if stacked.shape[0] != len(texts):
        raise RuntimeError(f"encoder returned {stacked.shape[0]} rows for {len(texts)} texts")
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        # A zero mean happens when the seed embeddings are
        # symmetric; the index has no direction to search. The
        # caller treats this like a cold start.
        return None
    result: NDArray[np.float32] = (mean / norm).astype(np.float32)
    return result


__all__ = [
    "SEED_TEXT_SEPARATOR",
    "resolve_query_vector",
]
