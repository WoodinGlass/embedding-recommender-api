"""Tests for the optional-import guard."""

from __future__ import annotations

import pytest

from recsys._optional import optional_import


def test_returns_module_when_present() -> None:
    mod = optional_import("json")
    assert mod is not None
    assert hasattr(mod, "dumps")


def test_returns_none_when_top_level_absent() -> None:
    mod = optional_import("this_package_does_not_exist_xyz")
    assert mod is None


def test_reraises_when_submodule_missing() -> None:
    # `json` exists but its submodule does not: broken install, not an
    # absent optional feature, so the error must propagate.
    with pytest.raises(ModuleNotFoundError):
        optional_import("json.nonexistent_submodule_xyz")


def test_faiss_absence_is_optional() -> None:
    # FAISS may or may not be installed; either way the guard must not raise.
    mod = optional_import("faiss")
    assert mod is None or hasattr(mod, "IndexFlatL2")
