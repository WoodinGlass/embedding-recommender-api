"""Unit tests for structured logging configuration.

The critical property: structlog output goes to **stderr**, not stdout.
CLI scripts print their final JSON result on stdout so a caller can
``json.load`` it. If logs landed on stdout as well, the stream would
contain log lines before the JSON and every consumer would fail with
"Extra data".
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import pytest

from recsys.config.enums import LogFormat
from recsys.config.settings import Settings
from recsys.monitoring import logging as log_mod

pytestmark = pytest.mark.unit


def _settings(**overrides: Any) -> Settings:
    # ``_env_file`` is a documented pydantic-settings runtime kwarg that
    # the type stubs do not expose. ``Any`` for the overrides is the
    # honest type at a **kwargs boundary that forwards to a validated
    # constructor: mypy cannot check the field types through **kwargs,
    # and the constructor validates them at runtime anyway.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_structlog_writes_to_stderr_not_stdout() -> None:
    settings = _settings(log_format=LogFormat.JSON)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        log_mod.configure_logging(settings)
        log = log_mod.get_logger("test")
        log.info("sentinel.event", key="value")

    assert "sentinel.event" in stderr_buf.getvalue(), stderr_buf.getvalue()
    assert "sentinel.event" not in stdout_buf.getvalue(), stdout_buf.getvalue()


def test_stdout_remains_clean_for_json_consumers() -> None:
    """A CLI that prints one JSON line after configure_logging must produce
    exactly that line on stdout."""
    import json

    settings = _settings(log_format=LogFormat.JSON)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        log_mod.configure_logging(settings)
        log = log_mod.get_logger("cli")
        log.info("cli.start")
        print(json.dumps({"event": "cli.done", "value": 42}))
        log.info("cli.end")

    lines = [ln for ln in stdout_buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1, stdout_buf.getvalue()
    doc = json.loads(lines[0])
    assert doc == {"event": "cli.done", "value": 42}


def test_logging_is_idempotent() -> None:
    settings = _settings()
    log_mod.configure_logging(settings)
    log_mod.configure_logging(settings)
    log = log_mod.get_logger("test")
    # No exception is the assertion.
    log.info("test.idempotent")


@pytest.fixture(autouse=True)
def _reset_root_logger() -> object:
    """Keep the stdlib root logger from leaking between tests."""
    root = logging.getLogger()
    before = list(root.handlers)
    yield None
    root.handlers[:] = before
