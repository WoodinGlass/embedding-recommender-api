"""Access log middleware — one structured line per request.

Emits the reserved fields from ``docs/contracts.md`` § 4.2 and records
duration / status / source metrics.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from recsys.monitoring.logging import get_logger
from recsys.monitoring.metrics import REQUEST_DURATION, REQUESTS_TOTAL

log = get_logger(__name__)

# Paths where an access log line is redundant or would add cardinality noise.
_EXCLUDED_PATHS: frozenset[str] = frozenset({"/metrics", "/healthz", "/readyz"})


def _path_template(request: Request) -> str:
    """Return the route template (``/v1/items/{item_id}/similar``).

    Never return the resolved path: raw path segments become Prometheus labels
    and every ``item_id`` would be a new time series.
    """
    route = request.scope.get("route")
    if route is None:
        return "unmatched"
    return getattr(route, "path", "unmatched") or "unmatched"


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Structured access log + Prometheus metrics for every request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            route = _path_template(request)
            # `source` defaults to `ann`; the recommend handler overrides it
            # via `request.state.source` once implemented in M3.
            source = getattr(request.state, "source", "ann")
            status_label = str(status)

            REQUESTS_TOTAL.labels(
                route=route,
                method=request.method,
                status=status_label,
                source=source,
            ).inc()
            REQUEST_DURATION.labels(
                route=route,
                source=source,
                status=status_label,
            ).observe(duration)

            log.info(
                "http.request",
                route=route,
                method=request.method,
                status=status,
                duration_ms=round(duration * 1000, 2),
                source=source,
            )
