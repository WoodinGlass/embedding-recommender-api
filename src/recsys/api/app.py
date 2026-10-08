"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from recsys import __version__
from recsys.api.middleware import (
    AccessLogMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from recsys.api.routers import churn, events, health, metrics, recommend
from recsys.config.enums import AppEnv
from recsys.config.hot import HotConfigError, HotConfigStore, default_hot_config
from recsys.config.settings import Settings, get_settings
from recsys.monitoring.logging import configure_logging, get_logger
from recsys.monitoring.tracing import configure_tracing
from recsys.rate_limit import TokenBucketLimiter
from recsys.resilience import CircuitBreaker


def _build_hot_store(settings: Settings, log: object) -> HotConfigStore:
    """Load the hot config; in prod a missing file is fatal.

    Outside prod the process starts with the built-in default so a
    fresh checkout (or a test that only cares about /healthz) does
    not need the file to exist.
    """
    path = Path(settings.hot_config_path)
    try:
        return HotConfigStore.from_path(path)
    except HotConfigError:
        if settings.app_env is AppEnv.PROD:
            raise
        log.warning("hot_config.fallback_to_default", path=str(path))  # type: ignore[attr-defined]
        return HotConfigStore(path, initial=default_hot_config())


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app.

    Called by the ASGI entrypoint (``recsys.api.app:app``) and by tests.
    All side-effecting setup (logging, tracing, shared clients) happens
    here so tests can opt out by passing their own settings instance.
    """
    settings = settings or get_settings()

    configure_logging(settings)
    configure_tracing(settings)
    log = get_logger(__name__)

    # The limiter and the breaker are built before the middleware stack
    # because ``add_middleware`` instantiates the middleware immediately.
    # The breaker is shared with the response cache (ADR-0015): Redis is
    # one dependency, and one breaker guards it. The lifespan only closes
    # the Redis connection on shutdown.
    #
    # The breaker's on_transition callback is wired to Prometheus in
    # M3.3.6; until then the transition events are emitted into the void.
    redis_breaker = CircuitBreaker(
        name="redis",
        failure_threshold=settings.redis_breaker_failure_threshold,
        open_seconds=settings.redis_breaker_open_seconds,
        open_max_seconds=settings.redis_breaker_open_max_seconds,
    )
    limiter = TokenBucketLimiter(
        redis_url=settings.redis_url,
        instance_count=settings.instance_count,
        bucket_seconds=settings.rate_limit_bucket_seconds,
        breaker=redis_breaker,
    )
    hot_store = _build_hot_store(settings, log)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.limiter = limiter
        app.state.redis_breaker = redis_breaker
        app.state.hot_config = hot_store
        try:
            yield
        finally:
            await limiter.close()

    app = FastAPI(
        title="embedding-recommender-api",
        version=__version__,
        description=(
            "Embedding-based recommendation service with low-latency ANN "
            "retrieval, re-ranking, and statistically valid A/B testing."
        ),
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Starlette wraps middleware in reverse order of `add_middleware`
    # calls: the last added is the outermost. The intended flow is
    # RequestContext -> RateLimit -> AccessLog -> router, so
    # RequestContext is added last and AccessLog first.
    app.add_middleware(AccessLogMiddleware)  # inner
    app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        limits=hot_store.get().rate_limit.as_class_map(),
        ip_limit_per_minute=settings.rate_limit_ip_per_minute,
        trusted_proxy_count=settings.trusted_proxy_count,
        credential_salt=settings.user_id_hash_salt,
    )
    app.add_middleware(RequestContextMiddleware)  # outer

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(recommend.router)
    app.include_router(events.router)
    app.include_router(churn.router)

    log.info("app.created", env=settings.app_env.value, version=__version__)
    return app


app = create_app()
