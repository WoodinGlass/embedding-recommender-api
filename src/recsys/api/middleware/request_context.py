"""Request context middleware.

Assigns or propagates a ``request_id``, binds it to structlog contextvars, and
echoes it back in the response header.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

import structlog
import structlog.contextvars
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Populate ``request.state.request_id`` and structlog contextvars."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        raw = request.headers.get(REQUEST_ID_HEADER, "")
        # Reject anything longer than 64 chars to bound the label.
        request_id = raw[:64] if raw else uuid.uuid4().hex[:16]

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
