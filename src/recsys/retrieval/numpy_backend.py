"""In-process exact kNN backend.

``NumpyBackend`` reads an embedding set into memory and answers each query
by computing cosine similarity against every row. It is O(N·d) per query
and O(N·d) of RAM, which is fine for the sample catalog (200 items × 384
floats ≈ 300 KB) and unusable for a million items. That is deliberate:
this backend is not a production path.

Its three uses are recorded in ADR-0012:

1. The exact kNN baseline in evaluation (ADR-0009). The ANN-fidelity
   metric requires a brute-force reference computed on the same vectors;
   ``NumpyBackend`` is that reference.
2. End-to-end evaluation in environments without pgvector. The primary
   development environment (Colab) cannot host a pgvector service; this
   backend lets the evaluation harness run there.
3. Regression tests comparing backend behavior. With ``ef_search`` large
   enough that HNSW degenerates to exact search, ``PgvectorBackend`` and
   ``NumpyBackend`` must agree (ADR-0012 § The two implementations must
   agree on exact search).

This backend is not registered in ``recsys.retrieval.registry`` and has no
``IndexBackendEnum`` value. It is constructed explicitly by evaluation and
by tests. Selecting it via ``INDEX_BACKEND=numpy`` is not supported and
must not be added — see ADR-0012 § Alternatives considered.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from recsys.embeddings import artifacts as art
from recsys.retrieval.filters import FILTER_FIELDS, validate_filters

#: Tolerance for the "is the query vector L2-normalized?" check. Cosine
#: similarity is only a valid score when both sides are unit vectors;
#: the caller is responsible for normalizing per ADR-0009 § 3. We do not
#: re-normalize silently — a wrong-norm query is a bug and should surface.
_NORM_TOLERANCE = 1e-3


class NumpyBackend:
    """Exact kNN over an in-memory numpy array.

    Satisfies :class:`recsys.retrieval.base.IndexBackend`. See the module
    docstring for what it is used for and what it is not.
    """

    name: str = "numpy"

    def __init__(
        self,
        *,
        item_ids: list[str],
        embeddings: NDArray[np.float32],
        item_metadata: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        """Construct from arrays.

        Prefer :meth:`from_run_directory` in production-adjacent code; this
        constructor is public so tests can build a backend from small
        synthetic arrays without touching disk.

        ``item_ids`` and ``embeddings`` must be aligned: row ``i`` of
        ``embeddings`` is the vector for ``item_ids[i]``. Rows are sorted
        by ``item_id`` internally so that the (unspecified but consistent)
        order among equal scores is stable across runs.

        ``item_metadata`` maps ``item_id`` to a string→string mapping of
        metadata fields (``category``, ``brand``, ``language``). It is
        required only if ``search`` is ever called with ``filters``;
        missing fields are treated as the empty string, so a filter on a
        field a given item lacks will not match it.
        """
        if len(item_ids) != embeddings.shape[0]:
            raise ValueError(
                f"item_ids has {len(item_ids)} entries but embeddings has "
                f"{embeddings.shape[0]} rows"
            )
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        if embeddings.dtype != np.float32:
            raise ValueError(f"embeddings must be float32, got {embeddings.dtype}")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item_ids contains duplicates")

        # Deterministic order: sort by item_id. M1 writes the Parquet in
        # this order already; sorting again is cheap and makes the
        # invariant unconditional.
        order = sorted(range(len(item_ids)), key=lambda i: item_ids[i])
        self._item_ids: list[str] = [item_ids[i] for i in order]
        self._embeddings: NDArray[np.float32] = embeddings[order]
        self._dim: int = int(embeddings.shape[1])
        self._index_of: dict[str, int] = {iid: i for i, iid in enumerate(self._item_ids)}

        # Pre-compute one numpy array of string values per filter field,
        # aligned with self._item_ids. Filter evaluation is then a
        # vectorized string comparison, which matters when N is large
        # enough that a per-item dict lookup would dominate the query.
        self._field_values: dict[str, NDArray[np.str_]] | None = None
        self._metadata_fields: frozenset[str] = frozenset()
        if item_metadata is not None:
            fields_present: set[str] = set()
            for iid in self._item_ids:
                fields_present.update(item_metadata.get(iid, {}).keys())
            self._metadata_fields = frozenset(fields_present)
            self._field_values = {
                field: np.asarray(
                    [item_metadata.get(iid, {}).get(field, "") for iid in self._item_ids],
                    dtype=np.str_,
                )
                for field in FILTER_FIELDS
            }

    # ----------------------------------------------------------------- props
    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the stored vectors."""
        return self._dim

    @property
    def size(self) -> int:
        """Number of indexed items."""
        return len(self._item_ids)

    @property
    def item_ids(self) -> list[str]:
        """The item ids, in the backend's internal (sorted) order."""
        return list(self._item_ids)

    def is_ready(self) -> bool:
        """A constructed backend is always ready.

        There is no external dependency to check: the vectors are in
        memory and the constructor either succeeded or raised.
        """
        return True

    def embedding_for(self, item_id: str) -> NDArray[np.float32] | None:
        """Return the stored vector for ``item_id``, or ``None`` if unknown.

        The evaluation runner uses this to build a query vector from the
        seed items of a golden query without re-reading the artifact tree.
        A lookup that misses returns ``None`` rather than raising: a
        missing seed is a data problem the caller may want to tolerate if
        at least one other seed is available (see
        ``recsys.evaluation.runner.encode_query``).
        """
        idx = self._index_of.get(item_id)
        if idx is None:
            return None
        row: NDArray[np.float32] = self._embeddings[idx, :]
        return row

    # ---------------------------------------------------------------- search
    def search(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(item_id, score)`` pairs.

        ``score`` is cosine similarity, clamped to ``[0.0, 1.0]`` to match
        ``docs/contracts.md`` § 0 and § 2.1. Clamping does not affect the
        ranking; it only makes the returned scores schema-valid. Cosine
        similarity between two L2-normalized vectors is mathematically in
        ``[-1, 1]``; in practice it is non-negative for sentence-embedding
        models, and any negative value is indistinguishable from "no
        similarity" for a ranking consumer.

        The returned list is sorted by score descending. The order among
        equal scores is the backend's internal order, which is item_id
        ascending by construction. Callers apply their own deterministic
        tie-break (ADR-0012); this backend's choice is convenient, not
        contractual.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if vector.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {vector.shape}")
        if vector.shape[0] != self._dim:
            raise ValueError(f"vector has dimension {vector.shape[0]}, expected {self._dim}")
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, 1.0, atol=_NORM_TOLERANCE):
            raise ValueError(
                f"query vector must be L2-normalized (|v| = {norm:.6f}); "
                f"normalizing is the caller's responsibility per ADR-0009 § 3"
            )

        normalized_filters = validate_filters(filters)

        # Optional filter mask.
        if normalized_filters is None:
            candidate_indices: NDArray[np.intp] | None = None
            candidate_ids: list[str] = self._item_ids
            candidate_embeddings: NDArray[np.float32] = self._embeddings
        else:
            mask = self._build_filter_mask(normalized_filters)
            candidate_indices = np.flatnonzero(mask)
            candidate_ids = [self._item_ids[i] for i in candidate_indices]
            candidate_embeddings = self._embeddings[candidate_indices]

        n_candidates = candidate_embeddings.shape[0]
        if n_candidates == 0:
            return []

        # Cosine similarity: both sides are unit vectors, so the dot
        # product is the cosine.
        similarities: NDArray[np.float32] = candidate_embeddings @ vector

        # Top-k selection. For k == n, a full sort is fastest. Otherwise,
        # argpartition gives O(n) selection, then we sort the k selected
        # scores to produce the correct order.
        k_eff = min(k, n_candidates)
        if k_eff == n_candidates:
            top_idx = np.argsort(-similarities, kind="stable")
        else:
            partitioned = np.argpartition(-similarities, k_eff - 1)[:k_eff]
            top_idx = partitioned[np.argsort(-similarities[partitioned], kind="stable")]

        # Build (item_id, score) pairs, clamped to the contract range.
        result: list[tuple[str, float]] = []
        for idx in top_idx:
            i = int(idx)
            score = float(similarities[i])
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0
            result.append((candidate_ids[i], score))
        return result

    # --------------------------------------------------------------- filters
    def search_with_vectors(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float, NDArray[np.float32]]]:
        """Same as :meth:`search`, with the candidate vector on each result.

        The vectors are already in memory, so the only extra cost is the
        copy of each returned row's array (a view into the embedding
        matrix; the caller may hold it past the next search, so a view is
        safe, but the return type is a plain array either way). MMR reads
        it and never mutates it.
        """
        pairs = self.search(vector=vector, k=k, filters=filters)
        result: list[tuple[str, float, NDArray[np.float32]]] = []
        for item_id, score in pairs:
            idx = self._index_of.get(item_id)
            if idx is None:
                # Unreachable: search() only returns ids that are in the
                # index. The guard keeps mypy from widening the type.
                continue
            row: NDArray[np.float32] = self._embeddings[idx, :]
            result.append((item_id, score, row))
        return result

    def _build_filter_mask(self, filters: Mapping[str, str]) -> NDArray[np.bool_]:
        """Return a boolean mask of items matching every filter.

        Raises ``ValueError`` if a filter field is not in the backend's
        metadata. The error is a programming error, not a user error: the
        API layer rejects unknown fields at ``422`` before they reach a
        backend (see ``docs/adr/0008-filter-strategy.md``).
        """
        if self._field_values is None:
            raise RuntimeError(
                "search() was called with filters but this backend was "
                "constructed without item_metadata"
            )
        for field in filters:
            if field not in self._metadata_fields:
                raise ValueError(
                    f"filter field {field!r} not present in item metadata; "
                    f"available: {sorted(self._metadata_fields)}"
                )
        mask = np.ones(len(self._item_ids), dtype=bool)
        for field, value in filters.items():
            mask &= self._field_values[field] == value
        return mask

    # ------------------------------------------------------------ factories
    @classmethod
    def from_run_directory(
        cls,
        run_dir: Path,
        *,
        item_metadata: Mapping[str, Mapping[str, str]] | None = None,
    ) -> NumpyBackend:
        """Load the embedding set from a completed run directory.

        ``run_dir`` is the directory containing ``embeddings.parquet``
        (i.e. ``artifacts/embeddings/runs/<run_id>/``). Rows are read
        batch-by-batch via :func:`recsys.embeddings.artifacts.iter_previous_batches`
        and concatenated once, which is the same reader the M1 pipeline
        uses to reuse unchanged rows. There is no second Parquet reader to
        keep in sync.
        """
        run_dir = Path(run_dir)
        parquet = run_dir / "embeddings.parquet"
        if not parquet.is_file():
            raise FileNotFoundError(f"no embeddings.parquet in {run_dir}")

        item_ids: list[str] = []
        chunks: list[NDArray[np.float32]] = []
        for batch in art.iter_previous_batches(parquet):
            item_ids.extend(batch.item_ids)
            chunks.append(batch.embeddings)
        if not chunks:
            raise ValueError(f"parquet has no rows: {parquet}")

        embeddings = np.concatenate(chunks, axis=0)
        return cls(
            item_ids=item_ids,
            embeddings=embeddings,
            item_metadata=item_metadata,
        )
