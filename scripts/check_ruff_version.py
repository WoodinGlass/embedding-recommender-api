#!/usr/bin/env python3
"""Enforce that the installed ruff matches the pinned version.

The project runs ruff in three environments:

- the developer's shell (``make lint``) — uses what
  ``pip install -e ".[dev-lite]"`` resolved;
- the pre-commit hook — uses an isolated environment the framework
  builds from ``.pre-commit-config.yaml``;
- CI — uses what ``pip install -e ".[dev-lite]"`` resolved.

When the three disagree, a commit can pass in one environment and
fail in another for reasons that have nothing to do with the code.
The M3.4 fixes for PT011 and PT018 are examples: the same rule can
exist in two ruff versions and still flag a different line depending
on what the linter learned in between.

This script asserts that the ruff **installed in the current
environment** matches the pin in ``pyproject.toml``. It does not
(cannot) check the hook environment; the hook uses the same pin by
construction, and a change to one without the other is caught in
review.

Exit codes: 0 if versions match, 1 otherwise (with a clear message
on stdout). Wired into ``make check`` and the CI lint job.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The pin is a plain string of the form "ruff==X.Y.Z" in the
# [project.optional-dependencies] dev-lite list.
_PIN_RE = re.compile(r"^ruff==(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")


def pinned_version() -> str:
    """Return the ruff version pinned in pyproject.toml, or raise."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = data.get("project", {}).get("optional-dependencies", {})
    dev_lite = extras.get("dev-lite")
    if not isinstance(dev_lite, list):
        raise RuntimeError("pyproject.toml: [project.optional-dependencies] dev-lite missing")
    for entry in dev_lite:
        m = _PIN_RE.match(str(entry).strip())
        if m is not None:
            return m.group("version")
    raise RuntimeError(
        "pyproject.toml: no 'ruff==X.Y.Z' pin found in dev-lite; "
        "the pin is what makes this check meaningful"
    )


def installed_version() -> str:
    """Return the version `python -m ruff --version` reports, or raise.

    The output format is stable across ruff releases: 'ruff X.Y.Z'.
    Parsing the last token is robust to a leading program name.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`python -m ruff --version` exited {result.returncode}; "
            f"stderr: {result.stderr.strip()[:200]}"
        )
    out = result.stdout.strip()
    # "ruff 0.16.10" -> "0.16.10"
    parts = out.split()
    if len(parts) < 2:
        raise RuntimeError(f"unexpected `ruff --version` output: {out!r}")
    return parts[-1]


def main() -> int:
    try:
        pinned = pinned_version()
    except (RuntimeError, tomllib.TOMLDecodeError, OSError) as exc:
        print(f"check_ruff_version: cannot read the pin: {exc}")
        return 1
    try:
        installed = installed_version()
    except RuntimeError as exc:
        print(f"check_ruff_version: cannot read the installed version: {exc}")
        return 1

    if pinned != installed:
        print(
            f"check_ruff_version: MISMATCH\n"
            f"  pyproject.toml pins ruff=={pinned}\n"
            f"  this environment has ruff {installed}\n"
            f"  reinstall the environment:\n"
            f'    pip install -e ".[dev-lite]"'
        )
        return 1

    print(f"check_ruff_version: OK (ruff {installed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
