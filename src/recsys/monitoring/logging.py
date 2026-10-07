"""Structured logging via :mod:`structlog`.

JSON logs in prod and staging, pretty console logs in dev. Every line carries
the reserved fields from ``docs/contracts.md`` § 4.2.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger

from recsys.config.enums import AppEnv, LogFormat
from recsys.config.settings import Settings

# ============================================================================
# Safe default configuration, applied at module import time.
#
# This runs before any caller has a chance to invoke configure_logging().
# The default that structlog ships with sends logs to **stdout**, which is
# the wrong stream: stdout is program output for CLI scripts (they print
# one JSON line and expect a consumer to parse it), and diagnostics belong
# on stderr. A CLI that only calls get_logger() -- without first calling
# configure_logging() -- would otherwise pollute stdout and produce errors
# like "Extra data: line 1 column 5 (char 4)" when a caller tries to parse
# the JSON.
#
# configure_logging() overrides this with the real settings; nothing about
# that function changes. The point is that the *default* is safe.
# ============================================================================
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
)


def configure_logging(settings: Settings) -> None:
    """Configure structlog and stdlib logging.

    Idempotent: safe to call more than once (e.g. per test app factory).
    """
    log_level = getattr(logging, settings.log_level.value)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format is LogFormat.JSON:
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=settings.app_env is AppEnv.DEV)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        # Logs go to stderr, not stdout. CLI scripts (scripts/build_index.py,
        # scripts/eval.py) print their final JSON result on stdout so a
        # caller can `json.load` it; if logs landed on stdout as well, the
        # stream would contain log lines before the JSON and every consumer
        # would fail with "Extra data". The rule is: stdout is program
        # output, stderr is diagnostics. See docs/embedding-pipeline.md § 6.
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a bound logger.

    Prefer this over ``structlog.get_logger()`` so the return type is stable
    for mypy. ``structlog.get_logger`` is typed as returning ``Any``; we
    narrow it here so callers get real type checking on log calls.

    The module-level ``structlog.configure`` call above ensures that even
    a caller who never invokes :func:`configure_logging` gets the safe
    default: JSON to **stderr**. That matters for CLI scripts (build_index,
    promote_index, rollback_index, eval) that print their final result to
    stdout; the structlog default would send log lines to the same stream
    and break any consumer that parses stdout as JSON.
    """
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    return cast(FilteringBoundLogger, logger)
