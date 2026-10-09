"""Run the API as a production-style ASGI server: ``python -m recsys``.

Autoreload is deliberately disabled here so the entry point is safe to use as
a container command. For local development with reload, run uvicorn directly:

    uvicorn recsys.api.app:app --reload
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Entry point for ``python -m recsys``."""
    uvicorn.run(
        # Factory target. `app.py` deliberately does not build the app
        # at module scope; the entrypoint is where the side effects
        # belong. See the note in that module for why.
        "recsys.api.app:create_app",
        factory=True,
        # Bind all interfaces: required for container ingress. Auth and rate
        # limiting are enforced by middleware, not by the bind address.
        host="0.0.0.0",  # noqa: S104
        port=8000,
        log_config=None,  # structlog handles formatting
        access_log=False,  # AccessLogMiddleware emits our structured line
        reload=False,
    )


if __name__ == "__main__":
    main()
