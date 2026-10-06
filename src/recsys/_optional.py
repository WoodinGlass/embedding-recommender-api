"""Optional dependency guards.

Optional packages (``faiss``, ``onnxruntime``, OTEL, ...) are imported through
:func:`optional_import` so a missing dependency degrades a feature instead of
crashing the whole service. See ``docs/decisions.md`` § 3 for the rationale.

Usage::

    from recsys._optional import optional_import

    faiss = optional_import("faiss")
    if faiss is None:
        raise RuntimeError(
            "FAISS backend requested but the 'bench' extra is not installed. "
            "Run: pip install -e '.[bench]'"
        )
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


def optional_import(module_name: str) -> ModuleType | None:
    """Import *module_name*, returning ``None`` if its top-level package is
    not installed.

    Missing *submodules* of an installed package (or missing transitive
    dependencies of an installed package) are **not** swallowed: those
    indicate a broken install, not an intentionally-absent optional feature.
    """
    top = module_name.partition(".")[0]
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    return importlib.import_module(module_name)
