"""FastAPI dependency providers.

The shared clients (limiter, cache store, breaker, hot config) live
on ``app.state`` and are set inside the app's lifespan. The
dependencies read them from the request; a handler that declares
``LimiterDep`` receives the same limiter the middleware uses.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from recsys.api.auth import (
    JwtError,
    Principal,
    validate_api_key,
    validate_jwt,
)
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


def _unauthorized(request: Request, message: str) -> HTTPException:
    """Build the 401 the auth dependency raises.

    The shape matches the handler's client errors (``detail`` is a
    dict with ``code``, ``message``, ``request_id``) so a client sees
    one envelope for every 4xx. The credential, when present, is
    never in the message: ADR-0013 forbids it.
    """
    request_id = str(getattr(request.state, "request_id", ""))
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": "unauthenticated",
            "message": message,
            "request_id": request_id,
        },
    )


def get_principal(request: Request) -> Principal:
    """Validate the request's credential and return a :class:`Principal`.

    Two credential models are accepted (ADR-0013):

    - ``X-API-Key: <key>`` — an Argon2id-hashed entry (or plaintext in
      dev). The key list lives in ``settings.api_key_set`` /
      ``settings.api_key_admin_set``.
    - ``Authorization: Bearer <jwt>`` — validated against
      ``settings.jwt_secret``, ``settings.jwt_algorithm``, and
      ``settings.jwt_max_age_seconds``.

    When both headers are present, the API key is tried first: it is
    the cheaper validation and the one the deployment configures by
    default. A request with neither, or with a credential that fails
    validation, is a 401.

    The settings come from ``request.app.state.settings`` rather than
    ``get_settings()`` because the app factory may have been built
    with an explicit ``Settings`` instance (tests, embedded callers);
    the dependency must see that instance, not the process default.
    """
    settings: Settings = request.app.state.settings

    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        principal = validate_api_key(
            api_key,
            read_entries=settings.api_key_set,
            admin_entries=settings.api_key_admin_set,
        )
        if principal is None:
            raise _unauthorized(request, "invalid API key")
        return principal

    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        try:
            return validate_jwt(
                token,
                secret=settings.jwt_secret,
                algorithm=settings.jwt_algorithm,
                max_age_seconds=settings.jwt_max_age_seconds,
            )
        except JwtError as exc:
            raise _unauthorized(request, f"invalid JWT: {exc.reason}") from None

    raise _unauthorized(request, "no credential presented")


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
PrincipalDep = Annotated[Principal, Depends(get_principal)]
