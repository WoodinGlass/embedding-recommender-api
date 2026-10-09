"""FastAPI application factory."""

from __future__ import annotations

import contextlib
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
from recsys.cache import CacheStore
from recsys.config.enums import AppEnv
from recsys.config.hot import HotConfigError, HotConfigStore, default_hot_config
from recsys.config.settings import Settings, get_settings
from recsys.experiments import (
    SUPPORTED_SCHEMA_VERSION,
    ExperimentsError,
    ExperimentsFile,
    load_experiments,
)
from recsys.monitoring.logging import configure_logging, get_logger
from recsys.monitoring.tracing import configure_tracing
from recsys.popularity import PopularityCache, PopularityCacheError
from recsys.rate_limit import TokenBucketLimiter
from recsys.resilience import CircuitBreaker, CircuitState
from recsys.resilience.metrics import (
    breaker_transition_callback,
    set_initial_state,
)


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


def _build_db_pool(settings: Settings) -> object:
    """Build a synchronous connection pool. Do not open it here.

    The pool is opened in the lifespan, where a failure is recoverable:
    a process that boots while PostgreSQL is down should start (its
    ``/healthz`` answers) and report not-ready via ``/readyz``, not
    refuse to boot. The readiness contract is ADR-0019; the pool's
    open is a check it reports, not a startup gate.

    ``psycopg_pool.ConnectionPool`` is the synchronous pool. ADR-0012
    keeps the retrieval backends synchronous; the handler is a plain
    ``def`` and FastAPI runs it in a thread pool, so a synchronous pool
    is the piece that matches. An async pool would force the handler to
    become async and every call inside it to be awaited, which is the
    ceremony ADR-0012 chose not to pay.
    """
    from psycopg_pool import ConnectionPool

    return ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=5,
        timeout=settings.db_breaker_timeout_seconds,
        open=False,
    )


def _build_popularity_cache(settings: Settings, log: object) -> PopularityCache:
    """Load the tier-4 popularity cache from disk.

    A missing file is expected on a fresh checkout that has not run
    ``make popularity-refresh`` yet; the cache is empty and tier 4 falls
    through to tier 5 (ADR-0020 § Tier 4). A malformed file is a
    different problem — it means a previous refresh wrote something
    wrong — and is logged at WARNING but does not abort startup, because
    tier 4 is a fallback: losing it degrades quality, not availability.

    The path is config-relative for the same reason the hot config and
    experiments file are: the container's WORKDIR holds the artifact
    directory tree.
    """
    path = Path(settings.popularity_cache_path)
    try:
        cache = PopularityCache.from_file(path)
        log.info("popularity.cache.loaded", path=str(path), size=cache.size)  # type: ignore[attr-defined]
        return cache
    except PopularityCacheError as exc:
        log.warning(  # type: ignore[attr-defined]
            "popularity.cache.missing_or_malformed",
            path=str(path),
            error=str(exc),
        )
        return PopularityCache.empty()


def _build_experiments(settings: Settings, log: object) -> ExperimentsFile:
    """Load ``experiments.yaml``; in prod a missing file is fatal.

    The behavior mirrors ``_build_hot_store``: a missing file outside
    prod falls back to an empty declaration so a fresh checkout (or a
    test that only cares about ``/healthz``) is not blocked by an
    optional file. A **malformed** file aborts startup in every
    environment: ADR-0017 § Validated at startup is explicit that a
    typo in the YAML is a startup error, not a fallback. The two cases
    are distinguished by ``Path.is_file()`` before ``load_experiments``
    runs, so a malformed file is never silently replaced by an empty
    declaration.
    """
    path = Path("experiments.yaml")
    if not path.is_file():
        if settings.app_env is AppEnv.PROD:
            raise ExperimentsError(
                f"experiments file not found in prod: {path}; "
                "commit experiments.yaml or set APP_ENV"
            )
        log.warning("experiments.fallback_to_empty", path=str(path))  # type: ignore[attr-defined]
        # Empty declaration, same shape the loader returns. The
        # schema version comes from the loader, not a hardcoded 1, so a
        # future bump is one constant rather than two.
        return ExperimentsFile(
            schema_version=SUPPORTED_SCHEMA_VERSION,
            experiments=(),
            path=path,
        )
    return load_experiments(path)


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

    # The limiter, the breaker, and the cache store are built before the
    # middleware stack because ``add_middleware`` instantiates the
    # middleware immediately. The breaker is shared by the limiter and
    # the cache (ADR-0015): Redis is one dependency, and one breaker
    # guards it. The lifespan only closes the Redis clients on shutdown.
    #
    # The breaker emits Prometheus metrics through the callback wired in
    # ``resilience.metrics``; the initial gauge value is set explicitly
    # so an alert on ``state != 0`` is unambiguous from the first scrape.
    redis_breaker = CircuitBreaker(
        name="redis",
        failure_threshold=settings.redis_breaker_failure_threshold,
        open_seconds=settings.redis_breaker_open_seconds,
        open_max_seconds=settings.redis_breaker_open_max_seconds,
        on_transition=breaker_transition_callback,
    )
    set_initial_state("redis", CircuitState.CLOSED)

    limiter = TokenBucketLimiter(
        redis_url=settings.redis_url,
        instance_count=settings.instance_count,
        bucket_seconds=settings.rate_limit_bucket_seconds,
        breaker=redis_breaker,
    )
    cache_store = CacheStore(
        redis_url=settings.redis_url,
        breaker=redis_breaker,
        socket_timeout_seconds=settings.cache_socket_timeout_seconds,
    )
    hot_store = _build_hot_store(settings, log)
    experiments_file = _build_experiments(settings, log)
    db_pool = _build_db_pool(settings)
    popularity_cache = _build_popularity_cache(settings, log)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.limiter = limiter
        app.state.cache = cache_store
        app.state.redis_breaker = redis_breaker
        app.state.hot_config = hot_store
        app.state.experiments = experiments_file
        app.state.db_pool = db_pool
        app.state.popularity_cache = popularity_cache
        # Open the pool here, not in ``_build_db_pool``: a failure to
        # connect is a readiness problem (ADR-0019), not a startup one.
        # The pool object stays on state even when its open fails; a
        # caller checks ``pool.closed`` before using it and falls
        # through the ADR-0020 tiers when the database is unreachable.
        try:
            db_pool.open(wait=True, timeout=settings.db_breaker_timeout_seconds)  # type: ignore[attr-defined]
        except Exception as exc:
            log.warning(
                "db.pool.open_failed",
                error=type(exc).__name__,
                message="continuing; readiness will report not-ready",
            )
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                db_pool.close()  # type: ignore[attr-defined]
            await limiter.close()
            await cache_store.close()

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


# NOTE: no `app = create_app()` at module scope. The factory is
# the entrypoint, not the built app: `uvicorn
# recsys.api.app:create_app --factory` calls it, the Dockerfile
# CMD (`python -m recsys`) calls it through `__main__`, and a
# test that imports this module just to reach `create_app` does
# not pay for the app being built as a side effect of the
# import. Building the app requires a database URL, a Redis URL,
# and the serving extras; those belong to the entrypoint, not to
# every module that touches `recsys.api`.
