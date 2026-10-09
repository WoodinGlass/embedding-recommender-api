"""FastAPI dependency providers.

The shared clients (limiter, cache store, breaker, hot config) live
on ``app.state`` and are set inside the app's lifespan. The
dependencies read them from the request; a handler that declares
``LimiterDep`` receives the same limiter the middleware uses.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from recsys.cache import CacheStore
from recsys.config.hot import HotConfigStore
from recsys.config.settings import Settings, get_settings
from recsys.experiments import ExperimentsFile
from recsys.popularity import PopularityCache
from recsys.rate_limit import TokenBucketLimiter
from recsys.resilience import CircuitBreaker


def get_request_id(request: Request) -> str:
    """Return the request_id assigned by ``RequestContextMiddleware``."""
    return str(getattr(request.state, "request_id", ""))


def get_limiter(request: Request) -> TokenBucketLimiter:
    """Return the process-wide rate limiter."""
    return request.app.state.limiter  # type: ignore[no-any-return]


def get_cache_store(request: Request) -> CacheStore:
    """Return the process-wide cache store."""
    return request.app.state.cache  # type: ignore[no-any-return]


def get_redis_breaker(request: Request) -> CircuitBreaker:
    """Return the breaker shared by the limiter and the cache."""
    return request.app.state.redis_breaker  # type: ignore[no-any-return]


def get_hot_config(request: Request) -> HotConfigStore:
    """Return the process-wide hot config store."""
    return request.app.state.hot_config  # type: ignore[no-any-return]


def get_experiments(request: Request) -> ExperimentsFile:
    """Return the process-wide experiments declaration."""
    return request.app.state.experiments  # type: ignore[no-any-return]


def get_db_pool(request: Request) -> object:
    """Return the process-wide synchronous connection pool.

    The return type is ``object`` at the dependency boundary: the
    concrete class lives behind the ``[db]`` extra and typing it
    precisely would pull the extra into the base environment. The
    caller casts (or passes the value to a protocol-typed argument)
    where it is used.
    """
    return request.app.state.db_pool


def get_popularity_cache(request: Request) -> PopularityCache:
    """Return the process-wide tier-4 in-memory popularity cache."""
    return request.app.state.popularity_cache  # type: ignore[no-any-return]


def get_encoder(request: Request) -> object | None:
    """Return the ONNX encoder, or ``None`` when it is not loaded.

    ``None`` is a normal state in dev and test (the artifact is derived
    and a fresh checkout has never run ``make export_onnx``); in prod
    the app factory refuses to boot without one (``_build_encoder``).
    """
    return getattr(request.app.state, "encoder", None)


SettingsDep = Annotated[Settings, Depends(get_settings)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
LimiterDep = Annotated[TokenBucketLimiter, Depends(get_limiter)]
CacheStoreDep = Annotated[CacheStore, Depends(get_cache_store)]
RedisBreakerDep = Annotated[CircuitBreaker, Depends(get_redis_breaker)]
HotConfigDep = Annotated[HotConfigStore, Depends(get_hot_config)]
ExperimentsDep = Annotated[ExperimentsFile, Depends(get_experiments)]
DbPoolDep = Annotated[object, Depends(get_db_pool)]
PopularityCacheDep = Annotated[PopularityCache, Depends(get_popularity_cache)]
EncoderDep = Annotated[object | None, Depends(get_encoder)]
