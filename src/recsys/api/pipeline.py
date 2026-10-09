"""Synchronous recommendation pipeline (ADR-0012, ADR-0016, ADR-0020).

The handler calls exactly one function of this module, once per
request, inside a bounded worker thread
(``anyio.to_thread.run_sync(..., limiter=CapacityLimiter(N))``;
ADR-0012 as amended in M3.6.6f). Everything downstream of the
handler entry — encode, retrieval, re-rank — runs here, so the
request pays one thread hop, not four. The function is synchronous
by design: the retrieval backend (ADR-0012) and the ONNX encoder
(ADR-0002) are synchronous.

**The result is a union, not a sentinel.** ``PipelineResult`` is a
successful ANN result; ``PipelineFailure`` carries a ``reason`` the
caller uses to decide the response. Two of the reasons are client
errors the caller turns into a ``4xx``; three are dependency
failures the caller turns into the fallback chain (ADR-0020).
Encoding the reason in the type is what keeps the mapping in one
place — the handler — and stops it from being re-derived,
differently, at each call site.

**Why not raise.** A pipeline failure is an expected outcome of the
request path, not a bug: an index that is being rebuilt, a Redis
outage, a catalog where every seed has been retired. Exceptions
model the unexpected; a union models the expected. The one
exception that can escape is a programming error inside the
pipeline itself, and the handler lets that one propagate to the
framework's 500 handler.

**Why providers are per-call.** ``PgRecencyProvider`` takes a
psycopg connection in its constructor (M3.6.3). Since the handler
gets a fresh connection from the pool per request, the recency
provider is constructed per request from that connection. It is a
small object that stores two pointers; the cost is negligible. A
future refactor could make the provider take a connection factory
and move it to the lifespan; that is a change to M3.6.3, not to
this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from recsys.retrieval.providers import PopularityProvider, RecencyProvider
from recsys.retrieval.rerank import Candidate, RerankConfig, Reranker

#: Why the pipeline did not produce a result.
#:
#: - ``empty_seeds`` — the request carried no seeds. Client error
#:   (``400``); a caller that wants popular items asks for them
#:   through the fallback path, not by sending an empty seed list.
#: - ``all_seeds_missing`` — every seed the request named is absent
#:   from the catalog. Client error (``404``); the client's catalog
#:   view is out of date.
#: - ``encoder_unavailable`` — the ONNX encoder is not loaded (only
#:   reachable in dev/test; prod refuses to boot without one). The
#:   caller moves to the fallback chain.
#: - ``no_active_index`` — no row in ``index_registry`` has
#:   ``status='active'``. The caller moves to the fallback chain.
#: - ``ann_error`` — the backend raised, or the re-ranker raised.
#:   The caller moves to the fallback chain.
PipelineFailureReason = Literal[
    "empty_seeds",
    "all_seeds_missing",
    "encoder_unavailable",
    "no_active_index",
    "ann_error",
]


@dataclass(frozen=True)
class PipelineResult:
    """A successful ANN result.

    ``items`` is a tuple of ``(item_id, score)`` pairs, already
    truncated to ``k`` and with the request's seeds excluded. The
    tuple (not list) is the same choice ``CacheResult`` makes: the
    value is a value, not a mutable buffer the caller can edit.

    ``index_version`` and ``model_version`` are what the response's
    ``meta`` carries; they are read from the backend and the encoder
    at the moment of the result, so they name the actual versions
    that produced the list.
    """

    items: tuple[tuple[str, float], ...]
    index_version: str
    model_version: str


@dataclass(frozen=True)
class PipelineFailure:
    """The pipeline did not produce a result; ``reason`` says why.

    ``detail`` is a short human-readable string for a log line. It
    is not returned to the client; the reason's category is what
    maps to an HTTP status (see the ``PipelineFailureReason``
    docstring).
    """

    reason: PipelineFailureReason
    detail: str | None = None


class _EncoderLike(Protocol):
    @property
    def model_version(self) -> str: ...
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class _BackendLike(Protocol):
    @property
    def active_index_version(self) -> str: ...
    def search(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]: ...
    def search_with_vectors(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float, NDArray[np.float32]]]: ...


def _safe_provider_get(
    provider: Any,
    ids: list[str],
) -> tuple[dict[str, float], bool]:
    """Call ``provider.get(ids)``; return ``({}, False)`` on failure.

    Same rule as ``evaluation.runner._safe_provider_get``: any
    exception is a missing signal. The provider contract is "raise"
    (ADR-0016 § Failure behavior): a timeout, a connection error, and
    a logic error are the same to the re-ranker.
    """
    if not ids:
        return {}, True
    try:
        result = provider.get(ids)
    except Exception:
        return {}, False
    return {str(k): float(v) for k, v in dict(result).items()}, True


def sync_pipeline(
    *,
    connection: Any,
    encoder: _EncoderLike | None,
    seed_item_ids: Sequence[str],
    k: int,
    filters: Mapping[str, str] | None,
    rerank_config: RerankConfig,
    reranker: Reranker,
    popularity_provider: PopularityProvider,
    recency_provider: RecencyProvider,
    hnsw_ef_search: int,
    active_index_version: str | None = None,
    pgvector_version: Any | None = None,
    backend: _BackendLike | None = None,
) -> PipelineResult | PipelineFailure:
    """Run the full synchronous pipeline for one recommend request.

    ``backend`` is optional: production passes ``None`` and the
    function builds a ``PgvectorBackend`` from the active index row;
    a unit test passes a fake. The distinction is what makes the
    pipeline testable without a database.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")

    if encoder is None:
        return PipelineFailure("encoder_unavailable")

    if not seed_item_ids:
        return PipelineFailure("empty_seeds")

    # Active index. ``from_registry`` reads ``index_registry`` and
    # raises ``RuntimeError`` if no row is active; that is the case
    # the fallback chain serves.
    if backend is None:
        from recsys.retrieval.pgvector import PgvectorBackend

        if active_index_version is None:
            return PipelineFailure("no_active_index", detail="active_index_version not provided")

        try:
            backend = PgvectorBackend(
                connection,
                active_index_version=active_index_version,
                hnsw_ef_search=hnsw_ef_search,
                pgvector_version=pgvector_version,
            )
        except RuntimeError as exc:
            return PipelineFailure("no_active_index", detail=str(exc)[:200])

    # Query vector. ``resolve_query_vector`` returns ``None`` when
    # every seed is absent from the catalog, or when the mean is the
    # zero vector (a symmetric seed set). Both are "no usable
    # query", which the client learns as a 4xx.
    from recsys.retrieval.query import resolve_query_vector

    query_vector = resolve_query_vector(
        connection, encoder=encoder, seed_item_ids=list(seed_item_ids)
    )
    if query_vector is None:
        return PipelineFailure("all_seeds_missing")

    # Retrieve. The candidate window is the multiplier the re-ranker
    # configuration fixes (ADR-0016 § Candidate window); the ANN call
    # is bounded by the caller's request timeout, not here.
    candidate_k = k * rerank_config.candidate_multiplier
    needs_vectors = rerank_config.enable_mmr and k >= rerank_config.mmr_min_k

    try:
        if needs_vectors:
            raw = backend.search_with_vectors(vector=query_vector, k=candidate_k, filters=filters)
            pairs: list[tuple[str, float]] = [(iid, float(s)) for iid, s, _ in raw]
            vectors: dict[str, NDArray[np.float32]] = {iid: v for iid, _, v in raw}
        else:
            pairs = backend.search(vector=query_vector, k=candidate_k, filters=filters)
            vectors = {}
    except Exception as exc:
        return PipelineFailure("ann_error", detail=f"{type(exc).__name__}: {exc}"[:200])

    if not pairs:
        # An empty retrieval is a valid answer from the backend (a
        # very selective filter). The chain does not have another
        # tier to reach for; the response is 200 with zero items.
        return PipelineResult(
            items=(),
            index_version=backend.active_index_version,
            model_version=encoder.model_version,
        )

    # Providers.
    ids = [iid for iid, _ in pairs]
    pop_map, pop_ok = _safe_provider_get(popularity_provider, ids)
    rec_map, rec_ok = _safe_provider_get(recency_provider, ids)
    missing_signals: set[str] = set()
    if not pop_ok:
        missing_signals.add("popularity")
    if not rec_ok:
        missing_signals.add("recency")

    candidates = [
        Candidate(
            item_id=iid,
            similarity=score,
            popularity=pop_map.get(iid, 0.0),
            age_days=rec_map.get(iid, 0.0),
            vector=vectors.get(iid),
        )
        for iid, score in pairs
    ]

    # Re-rank. The re-ranker must not add or remove items (its
    # protocol says so); it returns the same ids in a new order.
    # The seeds are removed *after* the re-rank, so the ranker sees
    # them and the top-k it returns is a candidate pool from which
    # the seeds are then dropped and the list is truncated to ``k``.
    seeds = set(seed_item_ids)
    try:
        ranked = reranker.rerank(
            candidates=candidates,
            k=k + len(seeds),
            missing_signals=frozenset(missing_signals),
        )
    except Exception as exc:
        return PipelineFailure("ann_error", detail=f"rerank: {type(exc).__name__}: {exc}"[:200])

    filtered = tuple((iid, score) for iid, score in ranked if iid not in seeds)[:k]
    return PipelineResult(
        items=filtered,
        index_version=backend.active_index_version,
        model_version=encoder.model_version,
    )


__all__ = [
    "PipelineFailure",
    "PipelineFailureReason",
    "PipelineResult",
    "sync_pipeline",
]
