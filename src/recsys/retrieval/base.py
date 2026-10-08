"""Retrieval backend interface.

Every ANN backend implements this protocol. The contract is intentionally
small: three members, one search method. Rationale for the shape — why it
is synchronous, why the query is an ``NDArray`` and not a Python list, why
the filter values are strings — is in
``docs/adr/0012-backend-abstraction.md``.

The tie-break rule (equal scores resolved by ``item_id`` ascending) is the
**caller's** responsibility, not the backend's. A backend returns results
sorted by score descending; ties are implementation-defined. See
``docs/retrieval-and-evaluation.md`` § 3.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class IndexBackend(Protocol):
    """Minimal contract every ANN backend satisfies.

    Implementations are not required to be thread-safe. The serving layer
    (M3) serializes access per backend instance.

    **Optional extension: ``search_with_vectors``.** A backend that can
    return the candidate vectors without a second round trip may
    implement ``search_with_vectors(...)`` with the same signature as
    :meth:`search` but returning ``(item_id, score, vector)`` triples.
    The re-ranker's MMR step needs those vectors; a backend without the
    method causes MMR to be skipped with
    ``recsys_rerank_skipped_total{reason="no_vectors"}`` (ADR-0016 §
    Step 3). The method is deliberately **not** on this protocol: a
    backend that does not implement it is still a valid backend, and
    listing it here would force every fake in every test to add a
    no-op. The caller detects the capability with
    ``hasattr(backend, "search_with_vectors")``.
    """

    name: str

    def is_ready(self) -> bool:
        """Return ``True`` when this backend can serve a search right now.

        A backend that cannot open its underlying store (no connection, no
        active index loaded) returns ``False``. The serving layer uses this
        to gate ``/readyz``; the evaluation harness does not (it calls
        :meth:`search` directly and lets exceptions propagate).
        """
        ...

    def search(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(item_id, score)`` pairs.

        - ``vector`` is a 1-D, L2-normalized ``float32`` array of length
          equal to the index's embedding dimension. The caller is
          responsible for normalizing it (see the mean-encoding rule in
          ``docs/adr/0009-golden-set-and-metrics.md`` § 3).
        - ``k`` is a positive integer. Fewer than ``k`` results may be
          returned if the filter is very selective.
        - ``filters`` is an optional mapping from filter field name to
          string value. Keys must be in ``FILTER_FIELDS``
          (``recsys.retrieval.filters``). Unknown keys are a programming
          error, not a user error — the API layer rejects them at ``422``
          before they reach a backend.

        The returned list is sorted by score descending. Ties are
        implementation-defined; the caller applies a deterministic
        tie-break.
        """
        ...
