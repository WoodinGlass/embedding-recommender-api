"""Run the API with uvicorn: ``python -m recsys``."""

from __future__ import annotations

import uvicorn

from recsys.config.settings import get_settings


def main() -> None:
    """Entry point for ``python -m recsys``."""
    settings = get_settings()
    uvicorn.run(
        "recsys.api.app:app",
        # Bind all interfaces: required for container ingress. Auth and rate
        # limiting are enforced by middleware, not by the bind address.
        host="0.0.0.0",  # noqa: S104
        port=8000,
        log_config=None,    # structlog handles formatting
        access_log=False,   # AccessLogMiddleware emits our structured line
        reload=settings.app_env.value == "dev",
    )


if __name__ == "__main__":
    main()
