"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from recsys import __version__
from recsys.api.middleware import AccessLogMiddleware, RequestContextMiddleware
from recsys.api.routers import churn, events, health, metrics, recommend
from recsys.config.settings import Settings, get_settings
from recsys.monitoring.logging import configure_logging, get_logger
from recsys.monitoring.tracing import configure_tracing


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app.

    Called by the ASGI entrypoint (``recsys.api.app:app``) and by tests. All
    side-effecting setup (logging, tracing) happens here so tests can opt out
    by passing their own settings instance.
    """
    settings = settings or get_settings()

    configure_logging(settings)
    configure_tracing(settings)
    log = get_logger(__name__)

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
    )

    # Starlette wraps middleware in reverse order of `add_middleware` calls:
    # the last added is the outermost. RequestContext must run first so the
    # access log can read `request.state.request_id`.
    app.add_middleware(AccessLogMiddleware)  # inner
    app.add_middleware(RequestContextMiddleware)  # outer

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(recommend.router)
    app.include_router(events.router)
    app.include_router(churn.router)

    log.info("app.created", env=settings.app_env.value, version=__version__)
    return app


app = create_app()
