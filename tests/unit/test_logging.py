"""Unit tests for structured logging configuration.

Every test in this module runs a small Python program in a **subprocess**
and inspects that subprocess's stdout and stderr. That is deliberate: an
in-process test using ``capsys``/``capfd`` cannot reliably observe what a
``PrintLoggerFactory`` writes, because structlog captures ``sys.stderr``
at the moment ``structlog.configure`` runs, and pytest's capturing
fixtures replace that stream *after* module import. A subprocess has a
clean start: whatever stream the logger writes to is a file descriptor
we can inspect directly.

The property under test is simple and load-bearing:

    structlog output goes to stderr, never to stdout.

CLI scripts (``scripts/build_index.py``, ``scripts/promote_index.py``,
``scripts/rollback_index.py``, ``scripts/eval.py``) print their final
result as a single JSON line on stdout so a caller can ``json.load`` it.
If logs landed on stdout as well, every consumer would fail with
"Extra data".
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

pytestmark = pytest.mark.unit


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with src/ on PYTHONPATH.

    The subprocess starts with a clean structlog configuration, so it
    observes whatever the imported module configures at import time,
    without interference from pytest's stream capture.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    # S603: the command is sys.executable plus a literal "-c", and the
    # program text is written by this test module — not user input, not
    # anything read from disk. The suppression documents that this call
    # site was considered.
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


# --------------------------------------------------------------------------- #
# module-level default
# --------------------------------------------------------------------------- #
def test_module_level_default_writes_to_stderr() -> None:
    """Importing the logging module and logging must not touch stdout.

    This is the property that makes CLI scripts safe: they import modules
    that call ``get_logger`` at import time, and if the module-level
    default wrote to stdout the CLI's JSON output would be corrupted.
    """
    code = (
        "from recsys.monitoring.logging import get_logger\n"
        "log = get_logger('test')\n"
        "log.info('default.event', key='v')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert "default.event" in result.stderr, f"log line not on stderr; stderr={result.stderr!r}"
    assert "default.event" not in result.stdout, (
        f"log line leaked to stdout; stdout={result.stdout!r}"
    )


def test_module_level_default_is_json() -> None:
    code = (
        "from recsys.monitoring.logging import get_logger\n"
        "log = get_logger('test')\n"
        "log.info('json.event', key='v')\n"
    )
    result = _run_python(code)
    line = [ln for ln in result.stderr.splitlines() if ln.strip()][-1]
    doc = json.loads(line)
    assert doc["event"] == "json.event"
    assert doc["key"] == "v"
    assert doc["level"] == "info"


# --------------------------------------------------------------------------- #
# configure_logging override
# --------------------------------------------------------------------------- #
def test_configure_logging_routes_to_stderr() -> None:
    code = (
        "from recsys.config.settings import Settings\n"
        "from recsys.monitoring import logging as lm\n"
        "lm.configure_logging(Settings(_env_file=None))\n"
        "log = lm.get_logger('test')\n"
        "log.info('configured.event', key='v')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert "configured.event" in result.stderr
    assert "configured.event" not in result.stdout


def test_stdout_remains_parseable_after_logging() -> None:
    """A CLI that prints one JSON line between two log calls produces
    exactly that line on stdout."""
    code = (
        "import json\n"
        "from recsys.monitoring.logging import get_logger\n"
        "log = get_logger('cli')\n"
        "log.info('cli.start')\n"
        "print(json.dumps({'event': 'cli.done', 'value': 42}))\n"
        "log.info('cli.end')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected 1 stdout line, got {len(lines)}: {result.stdout!r}"
    doc = json.loads(lines[0])
    assert doc == {"event": "cli.done", "value": 42}


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #
def test_configure_logging_is_idempotent() -> None:
    code = (
        "from recsys.config.settings import Settings\n"
        "from recsys.monitoring import logging as lm\n"
        "s = Settings(_env_file=None)\n"
        "lm.configure_logging(s)\n"
        "lm.configure_logging(s)\n"
        "log = lm.get_logger('test')\n"
        "log.info('idempotent.event')\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr
    assert "idempotent.event" in result.stderr
