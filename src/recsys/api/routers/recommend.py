"""Recommendation endpoints (ADR-0012, ADR-0016, ADR-0020).

The handler is ``async def`` and offloads the synchronous pipeline
(encode, retrieve, re-rank) to a bounded worker thread through
``anyio.to_thread.run_sync`` with a ``CapacityLimiter``. The
request pays one thread hop, not four; see ADR-0012 (amended in
M3.6.6f) for why the whole pipeline shares one hop and why the pool
is bounded.

The response is shaped per ``docs/contracts.md`` § 2.1. This commit
wires the happy path and the client-error branches; the fallback
chain (ADR-0020) and the cache lookup land in the next two
commits, and until then a dependency failure is a ``503``.
"""

from __future__ import annotations

from functools import partial

import anyio
from fastapi import APIRouter, HTTPException, Request, status

from recsys.api.deps import (
    CacheStoreDep,
    EncoderDep,
    ExperimentsDep,
    HotConfigDep,
    PrincipalDep,
    RequestIdDep,
    SettingsDep,
)
from recsys.api.pipeline import (
    PipelineFailure,
    PipelineResult,
    RetrievalOutcome,
    sync_rerank,
    sync_retrieve_and_rerank,
)
from recsys.api.schemas.recommend import (
    RecommendItem,
    RecommendMeta,
    RecommendRequest,
    RecommendResponse,
)
from recsys.cache import CacheLookup, CacheStore, build_cache_key, jittered_ttl
from recsys.fallback import FallbackResult, serve_fallback
from recsys.monitoring.logging import get_logger
from recsys.popularity import PopularityCache
from recsys.retrieval.providers import (
    PgRecencyProvider,
    SyntheticPopularityProvider,
)
from recsys.retrieval.rerank import Candidate, WeightedBlendReranker

router = APIRouter(prefix="/v1", tags=["recommend"])

log = get_logger(__name__)


def _to_response(
    *,
    request_id: str,
    source: str,
    model_version: str | None,
    index_version: str | None,
    items: tuple[tuple[str, float], ...],
) -> RecommendResponse:
    """Shape a response for the tier that produced ``items``.

    ``source`` and the version fields are the caller's: the ANN
    path passes ``"ann"`` with the versions from the pipeline;
    the fallback path passes its source with ``None`` versions
    (ADR-0020 — no model, no index is what a fallback is).
    """
    response_items = [
        RecommendItem(item_id=iid, score=float(score), rank=i + 1)
        for i, (iid, score) in enumerate(items)
    ]
    return RecommendResponse(
        request_id=request_id,
        items=response_items,
        meta=RecommendMeta(
            source=source,  # type: ignore[arg-type]
            model_version=model_version,
            index_version=index_version,
            experiment=None,
        ),
    )


async def _run_pipeline(
    request: Request,
    *,
    body: RecommendRequest,
    encoder: object | None,
    hot: object,
    pool: object,
    cache: CacheStore,
    limiter: anyio.CapacityLimiter,
    timeout_seconds: float,
) -> tuple[PipelineResult | PipelineFailure, str]:
    """Run retrieve + rerank, with a cache lookup before retrieve.

    Returns ``(result, source)`` where ``source`` is ``"ann"`` (the
    normal path, or the MMR-skip path) or ``"cache"`` (a hit). The
    handler uses ``source`` to fill ``meta.source``.

    Cache eligibility, from ADR-0015 amendment:

    - MMR-enabled requests skip the cache. A ``Candidate`` with its
      384-float vector is ~1.5 KB of JSON per row; caching it would
      bloat Redis for a config that is not the production default.
      The skip is counted with ``reason="mmr_enabled"``.
    - ``active_index`` must be known: the cache key includes it, and
      a missing one means the process never read the registry
      (database down at startup). The full path runs and the
      fallback chain handles the outcome.
    """
    settings: object = request.app.state.settings
    cfg = hot.get().rerank.to_rerank_config()  # type: ignore[attr-defined]
    filters = body.filters.model_dump(exclude_none=True) if body.filters is not None else None
    reranker = WeightedBlendReranker(cfg)

    active_index_version: str | None = getattr(request.app.state, "active_index", None)
    pgvector_version: object | None = getattr(request.app.state, "pgvector_version", None)

    cache_skip = cfg.enable_mmr or active_index_version is None
    if cfg.enable_mmr:
        from recsys.monitoring.metrics import CACHE_SKIP_TOTAL

        CACHE_SKIP_TOTAL.labels(reason="mmr_enabled").inc()

    cache_key: str | None = None
    if not cache_skip and active_index_version is not None:
        cache_key = build_cache_key(
            index_version=active_index_version,
            endpoint="recommend",
            k_max=hot.get().cache.cache_k_max,  # type: ignore[attr-defined]
            filters=filters,
            seed_item_ids=list(body.seed_item_ids),
            variant=None,
        )
        lookup = await cache.lookup(cache_key)
        if lookup.outcome is CacheLookup.HIT and lookup.items:
            cached = _candidates_from_cache(lookup.items)
            if cached:
                call = partial(
                    sync_rerank,
                    candidates=cached,
                    reranker=reranker,
                    k=body.k,
                    seed_item_ids=list(body.seed_item_ids),
                    index_version=active_index_version,
                    model_version=str(getattr(encoder, "model_version", "")),
                    # Empty on a hit: the retriever's record of which
                    # provider was down was not stored with the
                    # candidates. See the note in _candidates_from_cache.
                    missing_signals=frozenset(),
                )
                try:
                    with anyio.fail_after(timeout_seconds):
                        rerank_result: (
                            PipelineResult | PipelineFailure
                        ) = await anyio.to_thread.run_sync(call, limiter=limiter)
                except TimeoutError:
                    return (
                        PipelineFailure("ann_error", detail="rerank timeout on cache hit"),
                        "ann",
                    )
                return rerank_result, "cache"

    # Miss path (or MMR skip path): retrieve + rerank in one hop.
    try:
        pool_ctx = pool.connection()  # type: ignore[attr-defined]
    except Exception as exc:
        return PipelineFailure("ann_error", detail=f"pool unavailable: {type(exc).__name__}"), "ann"

    try:
        with pool_ctx as conn:
            recency_provider = PgRecencyProvider(conn, timeout_seconds=timeout_seconds)
            popularity_provider = SyntheticPopularityProvider()

            miss_call = partial(
                sync_retrieve_and_rerank,
                connection=conn,
                encoder=encoder,  # type: ignore[arg-type]
                seed_item_ids=list(body.seed_item_ids),
                k=body.k,
                filters=filters,
                rerank_config=cfg,
                reranker=reranker,
                popularity_provider=popularity_provider,
                recency_provider=recency_provider,
                hnsw_ef_search=settings.hnsw_ef_search,  # type: ignore[attr-defined]
                active_index_version=active_index_version,
                pgvector_version=pgvector_version,
            )
            try:
                with anyio.fail_after(timeout_seconds):
                    outcome: RetrievalOutcome = await anyio.to_thread.run_sync(
                        miss_call, limiter=limiter
                    )
            except TimeoutError:
                return (
                    PipelineFailure(
                        "ann_error", detail=f"pipeline timeout after {timeout_seconds}s"
                    ),
                    "ann",
                )
    except Exception as exc:
        return PipelineFailure("ann_error", detail=f"{type(exc).__name__}: {exc}"[:200]), "ann"

    # Store on the miss path when the retriever produced candidates.
    # Synchronous for M3.6 (the ADR-0015 § Cache-warming note about
    # a background task applies to the fire-and-forget variant; a
    # synchronous store is simpler to test and the latency is a
    # Redis round trip the request already pays). Background move is
    # tracked for M3.7.
    if cache_key is not None and outcome.candidates and isinstance(outcome.result, PipelineResult):
        rows = tuple(
            (c.item_id, c.similarity, c.popularity, c.age_days) for c in outcome.candidates
        )
        ttl = hot.get().cache.recommend_ttl_seconds  # type: ignore[attr-defined]
        await cache.store(cache_key, rows, ttl_seconds=jittered_ttl(ttl))

    return outcome.result, "ann"


def _candidates_from_cache(
    items: tuple[tuple[object, ...], ...],
) -> tuple[Candidate, ...]:
    """Rebuild ``Candidate`` values from cached four-tuples.

    A row that is not a four-tuple, or whose fields do not convert,
    drops the whole hit: a partially-reconstructed candidate list
    is worse than no cache. The handler treats an empty return as
    a miss.

    ``vector`` is ``None``: the ADR skips the cache for MMR, so a
    cache hit is by definition a non-MMR request and the re-ranker
    does not need it. ``missing_signals`` is not stored — a provider
    that was down during the miss replayed its zeros on the hit;
    the re-ranker's rank-based normalization gives uniform zeros
    one bin, so the ordering is preserved.
    """
    out: list[Candidate] = []
    for row in items:
        if len(row) != 4:
            return ()
        try:
            out.append(
                Candidate(
                    item_id=str(row[0]),
                    similarity=float(row[1]),  # type: ignore[arg-type]
                    popularity=float(row[2]),  # type: ignore[arg-type]
                    age_days=float(row[3]),  # type: ignore[arg-type]
                    vector=None,
                )
            )
        except (TypeError, ValueError):
            return ()
    return tuple(out)


def _sync_fallback(
    pool: object,
    popularity_cache: PopularityCache,
    *,
    k: int,
    filters: dict[str, str] | None,
) -> FallbackResult | None:
    """Run the tier-3 / tier-4 chain. Called on a worker thread.

    The connection is acquired here, not inside ``serve_fallback``,
    so its lifetime covers exactly the tier-3 read. When the pool
    is closed (the app booted with the database unreachable) or a
    connection cannot be acquired, ``serve_fallback`` runs with
    ``connection=None``: tier 3 is skipped and tier 4 serves from
    process memory. That is the degraded path ADR-0020 exists for.
    """
    try:
        pool_ctx = pool.connection()  # type: ignore[attr-defined]
    except Exception:
        return serve_fallback(
            connection=None,
            popularity_cache=popularity_cache,
            k=k,
            filters=filters,
        )
    try:
        with pool_ctx as conn:
            return serve_fallback(
                connection=conn,
                popularity_cache=popularity_cache,
                k=k,
                filters=filters,
            )
    except Exception:
        # The pool's context manager can raise on entry or exit
        # (a pool shutdown between connection() and __enter__).
        # Fall to tier 4 rather than 500.
        return serve_fallback(
            connection=None,
            popularity_cache=popularity_cache,
            k=k,
            filters=filters,
        )


async def _run_fallback(
    *,
    pool: object,
    popularity_cache: PopularityCache,
    limiter: anyio.CapacityLimiter,
    k: int,
    filters: dict[str, str] | None,
) -> FallbackResult | None:
    """Offload the synchronous fallback to a bounded worker thread.

    The tier-3 read is a database query and the tier-4 read is a
    comprehension over an in-memory list; neither belongs on the
    event loop. Both share the limiter the pipeline uses so a
    burst of fallbacks cannot starve the ANN path.
    """
    call = partial(
        _sync_fallback,
        pool,
        popularity_cache,
        k=k,
        filters=filters,
    )
    return await anyio.to_thread.run_sync(call, limiter=limiter)


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Top-k recommendations with metadata filters",
)
async def recommend(
    body: RecommendRequest,
    request: Request,
    principal: PrincipalDep,
    request_id: RequestIdDep,
    encoder: EncoderDep,
    hot: HotConfigDep,
    cache: CacheStoreDep,
    experiments: ExperimentsDep,
    settings: SettingsDep,
) -> RecommendResponse:
    """Return a ranked list for the request's seeds.

    The client-error branches (``400`` for an empty seed list,
    ``404`` for a seed list whose items are all absent) are the two
    ``PipelineFailure`` reasons that are the caller's to fix. The
    three dependency reasons are a ``503`` today; the fallback
    chain replaces that response in M3.6.6c.
    """
    del experiments  # used in 6d
    # `principal` is consumed by FastAPI before this function
    # runs: the auth dependency validates the credential. The
    # value is unused until M3.6.6d logs the exposure.
    del principal
    limiter: anyio.CapacityLimiter = request.app.state.thread_limiter

    result, source = await _run_pipeline(
        request,
        body=body,
        encoder=encoder,
        hot=hot,
        pool=request.app.state.db_pool,
        cache=cache,
        limiter=limiter,
        timeout_seconds=settings.request_timeout_seconds,
    )

    if isinstance(result, PipelineResult):
        log.info(
            "recommend.return",
            request_id=request_id,
            source=source,
            n_items=len(result.items),
            index_version=result.index_version,
        )
        return _to_response(
            request_id=request_id,
            source=source,
            model_version=result.model_version,
            index_version=result.index_version,
            items=result.items,
        )

    # Client errors.
    if result.reason == "empty_seeds":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "bad_request",
                "message": "seed_item_ids must be non-empty",
                "request_id": request_id,
            },
        )
    if result.reason == "all_seeds_missing":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "not_found",
                "message": "no seed item in the request exists in the catalog",
                "request_id": request_id,
            },
        )

    # Server-side failure. Try the fallback chain (ADR-0020
    # tiers 3 and 4) before giving up. Only server-side reasons
    # reach here: ``empty_seeds`` and ``all_seeds_missing`` were
    # raised above as client errors, and a client error must not
    # be turned into a popular list.
    fallback = await _run_fallback(
        pool=request.app.state.db_pool,
        popularity_cache=request.app.state.popularity_cache,
        limiter=limiter,
        k=body.k,
        filters=(body.filters.model_dump(exclude_none=True) if body.filters is not None else None),
    )
    if fallback is not None:
        log.info(
            "recommend.fallback",
            request_id=request_id,
            source=fallback.source,
            n_items=len(fallback.items),
            pipeline_reason=result.reason,
        )
        return _to_response(
            request_id=request_id,
            source=fallback.source,
            model_version=None,
            index_version=None,
            items=fallback.items,
        )

    # Every tier failed. This is the 503 from ADR-0020 § tier 5.
    log.warning(
        "recommend.unavailable",
        request_id=request_id,
        reason=result.reason,
        detail=result.detail,
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "unavailable",
            "message": f"no tier produced results: {result.reason}",
            "request_id": request_id,
        },
    )


@router.get(
    "/items/{item_id}/similar",
    response_model=RecommendResponse,
    summary="Item-to-item similarity",
)
async def similar(item_id: str, principal: PrincipalDep) -> RecommendResponse:
    del principal  # auth only; the handler is a stub until M3.6.6e
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Item-to-item similarity for {item_id!r} lands in M3.6.6e.",
    )
