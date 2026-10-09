"""Retrieval package.

**Deliberately empty of imports.** ``base.py`` imports numpy, and
any import of ``recsys.retrieval.<anything>`` triggers this
``__init__`` first. A consumer that only needs
``recsys.retrieval.filters`` (a numpy-free allowlist) would otherwise
pull numpy in transitively. Import the submodule you need directly:

    from recsys.retrieval.base import IndexBackend
    from recsys.retrieval.registry import register_backend
    from recsys.retrieval.filters import FILTER_FIELDS

A re-export here is a small convenience that costs an extra
dependency at import time; the cost is real (see the M3.6 CI
incidents) and the convenience is not.
"""
