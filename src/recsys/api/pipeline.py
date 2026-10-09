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


@dataclass(frozen=True)
class RetrievalResult:
    """Candidates the retriever produced, before re-ranking.

    ``sync_retrieve`` returns this; ``sync_rerank`` consumes it.
    Splitting the pipeline is what makes the cache possible: the
    cache stores retrieval, not the re-ranked list (ADR-0015), so a
    cache hit skips retrieve and still runs rerank.

    ``candidates`` is a tuple for the same reason ``PipelineResult.
    items`` is: a value, not a mutable buffer. ``index_version`` and
    ``model_version`` travel with the candidates so ``sync_rerank``
    can stamp the response without asking the backend again.
    """

    candidates: tuple[Candidate, ...]
    index_version: str
    model_version: str
    #: Provider signals that were missing when the candidates were
    #: built. Travels with the result so ``sync_rerank`` (and the
    #: cache-hit path) can pass it to the re-ranker. A cache hit has
    #: candidates but no provider call; the re-ranker still needs to
    #: know which signals were unavailable.
    missing_signals: frozenset[str] = frozenset()


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


def sync_retrieve(
    *,
    connection: Any,
    encoder: _EncoderLike | None,
    seed_item_ids: Sequence[str],
    k: int,
    filters: Mapping[str, str] | None,
    rerank_config: RerankConfig,
    popularity_provider: PopularityProvider,
    recency_provider: RecencyProvider,
    hnsw_ef_search: int,
    active_index_version: str | None = None,
    pgvector_version: Any | None = None,
    backend: _BackendLike | None = None,
) -> RetrievalResult | PipelineFailure:
    """Encode, resolve the query, retrieve, and build candidates.

    Everything up to (and not including) the re-rank. The re-ranker
    runs in ``sync_rerank`` because the cache (ADR-0015) stores the
    candidate list, not the re-ranked response — a change to the
    rerank config invalidates the response, not the retrieval.

    Returns ``RetrievalResult`` when the retriever produced a list
    (which may be empty: a very selective filter is a legitimate
    answer) or ``PipelineFailure`` for every reason the old
    ``sync_pipeline`` returned before reranking.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")

    if encoder is None:
        return PipelineFailure("encoder_unavailable")

    if not seed_item_ids:
        return PipelineFailure("empty_seeds")

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

    from recsys.retrieval.query import resolve_query_vector

    query_vector = resolve_query_vector(
        connection, encoder=encoder, seed_item_ids=list(seed_item_ids)
    )
    if query_vector is None:
        return PipelineFailure("all_seeds_missing")

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
        return RetrievalResult(
            candidates=(),
            index_version=backend.active_index_version,
            model_version=encoder.model_version,
        )

    ids = [iid for iid, _ in pairs]
    pop_map, pop_ok = _safe_provider_get(popularity_provider, ids)
    rec_map, rec_ok = _safe_provider_get(recency_provider, ids)
    missing_signals: set[str] = set()
    if not pop_ok:
        missing_signals.add("popularity")
    if not rec_ok:
        missing_signals.add("recency")

    candidates = tuple(
        Candidate(
            item_id=iid,
            similarity=score,
            popularity=pop_map.get(iid, 0.0),
            age_days=rec_map.get(iid, 0.0),
            vector=vectors.get(iid),
        )
        for iid, score in pairs
    )
    # ``missing_signals`` travels on the result, not as a side channel:
    # a cache hit has candidates but no provider call, and the re-ranker
    # still needs to know the signals' health. Stored as the same
    # frozenset the re-ranker expects.
    return RetrievalResult(
        candidates=candidates,
        index_version=backend.active_index_version,
        model_version=encoder.model_version,
        missing_signals=frozenset(missing_signals),
    )


def sync_rerank(
    *,
    candidates: Sequence[Candidate],
    reranker: Reranker,
    k: int,
    seed_item_ids: Sequence[str],
    index_version: str,
    model_version: str,
    missing_signals: frozenset[str] = frozenset(),
) -> PipelineResult | PipelineFailure:
    """Re-rank the candidates and shape the response items.

    Seeds are removed *after* the re-rank: the ranker sees them, and
    the top-k it returns is a candidate pool from which the seeds
    are dropped and the list is truncated to ``k``. Removing before
    would tell the ranker to ignore a signal the caller asked for.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")

    if not candidates:
        return PipelineResult(
            items=(),
            index_version=index_version,
            model_version=model_version,
        )

    seeds = set(seed_item_ids)
    try:
        ranked = reranker.rerank(
            candidates=list(candidates),
            k=k + len(seeds),
            missing_signals=missing_signals,
        )
    except Exception as exc:
        return PipelineFailure("ann_error", detail=f"rerank: {type(exc).__name__}: {exc}"[:200])

    filtered = tuple((iid, score) for iid, score in ranked if iid not in seeds)[:k]
    return PipelineResult(
        items=filtered,
        index_version=index_version,
        model_version=model_version,
    )


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
    """Run retrieve then rerank as one call.

    Kept for callers that do not need the split (the evaluation
    runner, the load test's warmup). The handler uses the split
    so the cache can sit between the two halves.
    """
    retrieval = sync_retrieve(
        connection=connection,
        encoder=encoder,
        seed_item_ids=seed_item_ids,
        k=k,
        filters=filters,
        rerank_config=rerank_config,
        popularity_provider=popularity_provider,
        recency_provider=recency_provider,
        hnsw_ef_search=hnsw_ef_search,
        active_index_version=active_index_version,
        pgvector_version=pgvector_version,
        backend=backend,
    )
    if isinstance(retrieval, PipelineFailure):
        return retrieval

    return sync_rerank(
        candidates=retrieval.candidates,
        reranker=reranker,
        k=k,
        seed_item_ids=seed_item_ids,
        index_version=retrieval.index_version,
        model_version=retrieval.model_version,
        missing_signals=retrieval.missing_signals,
    )


@dataclass(frozen=True)
class RetrievalOutcome:
    """Retrieve+rerank in one call, with the candidates the handler caches.

    ``result`` is what the response carries. ``candidates`` is what
    the handler serializes into the cache: present only when the
    retrieve step produced a non-empty candidate list. A retrieval
    that failed before the backend, or produced zero candidates,
    carries ``candidates=None`` — the handler does not cache a
    failure or an empty list.

    ``index_version`` and ``model_version`` travel with the
    candidates because the cache value has no room for them (the
    row shape is id/sim/pop/age, the ADR-0015 § cache schema); the
    handler reads them off the response when the pipeline succeeds.
    """

    result: PipelineResult | PipelineFailure
    candidates: tuple[Candidate, ...] | None


def sync_retrieve_and_rerank(
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
) -> RetrievalOutcome:
    """Run retrieve then rerank; return both the response and the candidates.

    The handler calls this on the miss path (or the MMR skip path).
    The candidates it returns are what the handler serializes into
    the cache: the retrieval output, not the re-ranked list. Caching
    the re-ranked list would serve a stale ordering after a rerank
    config change (ADR-0015 § Do not cache the re-ranked result).
    """
    retrieval = sync_retrieve(
        connection=connection,
        encoder=encoder,
        seed_item_ids=seed_item_ids,
        k=k,
        filters=filters,
        rerank_config=rerank_config,
        popularity_provider=popularity_provider,
        recency_provider=recency_provider,
        hnsw_ef_search=hnsw_ef_search,
        active_index_version=active_index_version,
        pgvector_version=pgvector_version,
    )
    if isinstance(retrieval, PipelineFailure):
        return RetrievalOutcome(result=retrieval, candidates=None)

    rerank = sync_rerank(
        candidates=retrieval.candidates,
        reranker=reranker,
        k=k,
        seed_item_ids=seed_item_ids,
        index_version=retrieval.index_version,
        model_version=retrieval.model_version,
        missing_signals=retrieval.missing_signals,
    )
    return RetrievalOutcome(
        result=rerank,
        candidates=retrieval.candidates if retrieval.candidates else None,
    )


__all__ = [
    "PipelineFailure",
    "PipelineFailureReason",
    "PipelineResult",
    "RetrievalOutcome",
    "RetrievalResult",
    "sync_pipeline",
    "sync_rerank",
    "sync_retrieve",
    "sync_retrieve_and_rerank",
]
