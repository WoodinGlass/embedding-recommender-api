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
    sync_pipeline,
)
from recsys.api.schemas.recommend import (
    RecommendItem,
    RecommendMeta,
    RecommendRequest,
    RecommendResponse,
)
from recsys.monitoring.logging import get_logger
from recsys.retrieval.providers import (
    PgRecencyProvider,
    SyntheticPopularityProvider,
)
from recsys.retrieval.rerank import WeightedBlendReranker

router = APIRouter(prefix="/v1", tags=["recommend"])

log = get_logger(__name__)


def _to_response(
    *,
    request_id: str,
    result: PipelineResult,
) -> RecommendResponse:
    """Turn a ``PipelineResult`` into the response schema."""
    items = [
        RecommendItem(item_id=iid, score=float(score), rank=i + 1)
        for i, (iid, score) in enumerate(result.items)
    ]
    return RecommendResponse(
        request_id=request_id,
        items=items,
        meta=RecommendMeta(
            source="ann",
            model_version=result.model_version,
            index_version=result.index_version,
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
    limiter: anyio.CapacityLimiter,
    timeout_seconds: float,
) -> PipelineResult | PipelineFailure:
    """Build the per-request collaborators and run ``sync_pipeline``.

    The collaborators that depend on a live connection are built
    here, inside the pool's context manager, so the connection's
    lifetime covers the pipeline call exactly. The re-ranker is
    built per-request from the hot config so a config change takes
    effect without a restart (ADR-0022); the re-ranker object itself
    is thin (it stores the providers and the config, no I/O).

    The timeout is ``anyio.fail_after``, not a per-call database
    timeout: the whole pipeline shares one wall-clock budget, and a
    request that exceeds it is a fallback signal, not a partial
    result. Cancelling ``to_thread.run_sync`` does not stop the
    thread (the thread keeps running to completion); the request
    returns immediately and the thread's result is discarded. See
    the ADR-0012 amendment for the accepted trade-off.
    """
    # The pool is closed when the process booted with a database
    # that is unreachable; the failure is a fallback signal, not a
    # 500 (the fallback chain is 6c; today it becomes a 503).
    try:
        pool_ctx = pool.connection()  # type: ignore[attr-defined]
    except Exception as exc:
        return PipelineFailure("ann_error", detail=f"pool unavailable: {type(exc).__name__}")

    try:
        with pool_ctx as conn:
            cfg = hot.get().rerank.to_rerank_config()  # type: ignore[attr-defined]
            reranker = WeightedBlendReranker(cfg)
            popularity_provider = SyntheticPopularityProvider()
            recency_provider = PgRecencyProvider(conn, timeout_seconds=timeout_seconds)
            settings = request.app.state.settings  # see app.py

            call = partial(
                sync_pipeline,
                connection=conn,
                encoder=encoder,  # type: ignore[arg-type]
                seed_item_ids=list(body.seed_item_ids),
                k=body.k,
                filters=(
                    body.filters.model_dump(exclude_none=True) if body.filters is not None else None
                ),
                rerank_config=cfg,
                reranker=reranker,
                popularity_provider=popularity_provider,
                recency_provider=recency_provider,
                hnsw_ef_search=settings.hnsw_ef_search,
            )
            try:
                # ``anyio.fail_after`` is a *sync* context manager in
                # anyio 4.x: it returns a ``CancelScope`` you enter with
                # ``with``. ``async with`` would raise ``AttributeError:
                # __aenter__`` at runtime. The scope still cancels the
                # ``await`` inside it, which is what we want.
                with anyio.fail_after(timeout_seconds):
                    return await anyio.to_thread.run_sync(call, limiter=limiter)
            except TimeoutError:
                return PipelineFailure(
                    "ann_error",
                    detail=f"pipeline timeout after {timeout_seconds}s",
                )
    except Exception as exc:
        return PipelineFailure("ann_error", detail=f"{type(exc).__name__}: {exc}"[:200])


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
    del cache, experiments  # used in 6c/6d; declared now for a stable signature
    # `principal` is consumed by FastAPI before this function
    # runs: the auth dependency validates the credential. The
    # value is unused until M3.6.6d logs the exposure.
    del principal
    limiter: anyio.CapacityLimiter = request.app.state.thread_limiter

    result = await _run_pipeline(
        request,
        body=body,
        encoder=encoder,
        hot=hot,
        pool=request.app.state.db_pool,
        limiter=limiter,
        timeout_seconds=settings.request_timeout_seconds,
    )

    if isinstance(result, PipelineResult):
        log.info(
            "recommend.ann",
            request_id=request_id,
            n_items=len(result.items),
            index_version=result.index_version,
        )
        return _to_response(request_id=request_id, result=result)

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

    # Dependency errors. The fallback chain replaces this 503 in
    # M3.6.6c; the reason is logged so the interim behavior is
    # traceable.
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
