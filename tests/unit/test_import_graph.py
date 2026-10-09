"""Import-graph contract tests.

The M3.6 milestone spent six CI reds on the same failure: a module
that should not need a heavy dependency pulled it in transitively,
and the failure surfaced as "the whole package cannot import" in
every environment that lacked the dependency.

These tests make the property explicit. Each one runs a **fresh
Python process** — not the current interpreter — and asserts that
the module imports without the forbidden dependencies appearing in
``sys.modules``. A subprocess is required because pytest itself has
already imported numpy, psycopg, and redis by the time the tests
run; only a clean process can answer "does this module pull X?".

The cost is a few subprocess spawns per test session (each is tens
of milliseconds); the benefit is a CI signal at commit time instead
of after a full test run.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _imports_forbidden(module: str, forbidden: tuple[str, ...]) -> None:
    """Import ``module`` in a fresh process; fail if any forbidden
    top-level package is loaded."""
    forbidden_repr = repr(list(forbidden))
    code = (
        "import sys\n"
        f"import {module}\n"
        f"forbidden = set({forbidden_repr})\n"
        "loaded = set(sys.modules)\n"
        "leaked = sorted(loaded & forbidden)\n"
        "if leaked:\n"
        "    sys.stderr.write(\n"
        "        f'{module} pulled forbidden modules: {leaked}\\n'\n"
        "    )\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"importing {module!r} pulled a forbidden dependency:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------- #
# numpy must not leak into schemas, filters, or the popularity cache
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module",
    [
        "recsys.api.schemas.events",
        "recsys.api.schemas.recommend",
        "recsys.api.schemas.common",
        "recsys.retrieval.filters",
        "recsys.popularity.cache",
        "recsys.popularity.snapshot",
        "recsys.experiments.assignment",
        "recsys.experiments.loader",
        "recsys.events.hashing",
    ],
)
def test_light_module_does_not_pull_numpy(module: str) -> None:
    _imports_forbidden(module, ("numpy",))


# ---------------------------------------------------------------- #
# psycopg must not leak into the API schemas
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module",
    [
        "recsys.api.schemas.events",
        "recsys.api.schemas.recommend",
    ],
)
def test_api_schemas_do_not_pull_psycopg(module: str) -> None:
    _imports_forbidden(module, ("psycopg", "psycopg_pool"))


# ---------------------------------------------------------------- #
# redis must not leak into the popularity cache or the schemas
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module",
    [
        "recsys.api.schemas.events",
        "recsys.popularity.cache",
    ],
)
def test_module_does_not_pull_redis(module: str) -> None:
    _imports_forbidden(module, ("redis",))


# ---------------------------------------------------------------- #
# api package does not build the app at import
# ---------------------------------------------------------------- #
def test_recsys_api_import_does_not_build_the_app() -> None:
    """Importing ``recsys.api`` must not call ``create_app``.

    The call is the entrypoint's job; the side effect at import time
    is what made six test modules fail collection when the serving
    extras were missing. If this test fails, someone put
    ``app = create_app()`` back at module scope or re-exported
    ``app`` from ``recsys.api.__init__``.
    """
    code = (
        "import sys\n"
        "import recsys.api\n"
        "# The pool is built by create_app; if it was built, the\n"
        "# recsys.popularity package was imported and psycopg came\n"
        "# with it.\n"
        "forbidden = {'psycopg', 'psycopg_pool', 'numpy'}\n"
        "leaked = sorted(set(sys.modules) & forbidden)\n"
        "if leaked:\n"
        "    sys.stderr.write(\n"
        "        f'recsys.api import pulled: {leaked}\\n'\n"
        "    )\n"
        "    sys.exit(1)\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        "importing recsys.api pulled the serving-path dependencies:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_retrieval_init_does_not_pull_numpy() -> None:
    """Importing the ``recsys.retrieval`` package must not pull numpy.

    The package's ``__init__`` is deliberately empty; a re-export of
    ``IndexBackend`` (which imports numpy) would make every consumer
    of ``recsys.retrieval.filters`` — a numpy-free allowlist — pull
    numpy in transitively.
    """
    _imports_forbidden("recsys.retrieval", ("numpy",))
