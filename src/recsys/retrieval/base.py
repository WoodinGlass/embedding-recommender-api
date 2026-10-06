"""Retrieval backend interface.

Concrete backends (pgvector, FAISS) implement this protocol. The registry in
:mod:`recsys.retrieval.registry` resolves an :class:`IndexBackend` enum value
to an implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexBackend(Protocol):
    """Minimal contract every ANN backend must satisfy."""

    name: str

    def is_ready(self) -> bool:
        """Return True when the backend has an active index loaded."""
        ...

    async def search(
        self,
        *,
        vector: list[float],
        k: int,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(item_id, score)`` pairs sorted by score descending."""
        ...
