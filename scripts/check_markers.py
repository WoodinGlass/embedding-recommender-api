#!/usr/bin/env python3
"""Enforce test-tier discipline.

Rules
-----
``tests/unit/**`` must not reference ``pytest.mark.integration`` or
``pytest.mark.load``. Unit tests belong to the fast tier; if a test needs an
external service, it belongs in ``tests/integration``.

``tests/integration/**`` and ``tests/load/**`` tier markers are applied
automatically by ``tests/conftest.py`` — no source check is needed here.

Exit code 0 if all files comply; 1 with a report otherwise. Wired into
``make check`` and CI.
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"

FORBIDDEN_IN_UNIT: tuple[str, ...] = (
    "pytest.mark.integration",
    "pytest.mark.load",
)


def iter_test_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return every ``test_*.py`` file under *root*, sorted for stable output."""
    return sorted(p for p in root.rglob("test_*.py") if p.is_file())


def check_file(path: pathlib.Path) -> list[str]:
    """Return a list of human-readable violations for *path* (empty if clean)."""
    rel = path.relative_to(REPO_ROOT)
    tier = path.relative_to(TESTS_ROOT).parts[0]
    if tier != "unit":
        return []

    src = path.read_text(encoding="utf-8")
    return [
        f"{rel}: {needle} must not appear under tests/unit/ "
        f"(move the test to tests/integration/ instead)"
        for needle in FORBIDDEN_IN_UNIT
        if needle in src
    ]


def main() -> int:
    if not TESTS_ROOT.is_dir():
        print(f"error: {TESTS_ROOT} not found", file=sys.stderr)
        return 1

    violations: list[str] = []
    for path in iter_test_files(TESTS_ROOT):
        violations.extend(check_file(path))

    if violations:
        print("Test-marker discipline violations:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print("OK: test-marker discipline holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
