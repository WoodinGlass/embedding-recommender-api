"""Registry for retrieval backends.

Maps an ``IndexBackendEnum`` value to the **class** that implements it.
The registry is a class registry, not a factory registry: a backend class
may require arguments (a database connection, a run directory) and the
registry does not know how to supply them. Callers instantiate the class
with the arguments their context provides.

Production code rarely consults the registry. Its two uses are:

1. **Test injection** — a test can substitute a fake backend for the
   production one without patching an import.
2. **Future plugin points** — if the project ever ships a third backend,
   one registration line makes it discoverable via ``INDEX_BACKEND``.

The M0 scaffold modeled this as a factory registry with zero-argument
factories. That model breaks for ``PgvectorBackend``, which needs a
connection, and for ``NumpyBackend``, which needs a run directory. ADR-0012
records the shape backends actually have. A class registry matches that
shape; a factory registry does not.

The default registrations are:

- ``pgvector`` → :class:`recsys.retrieval.pgvector.PgvectorBackend`
- ``faiss`` → a class whose constructor raises. FAISS is a benchmark-only
  path (ADR-0011); running it through the registry is a mistake and the
  error says so.

``NumpyBackend`` is deliberately not registered. It is not a production
backend (ADR-0012); selecting it via ``INDEX_BACKEND=numpy`` is not
supported. Construct it directly, as evaluation and tests do.
"""

from __future__ import annotations

from typing import Any

from recsys.config.enums import IndexBackend as IndexBackendEnum
from recsys.monitoring.logging import get_logger
from recsys.retrieval.base import IndexBackend

log = get_logger(__name__)


class _FaissNotAProductionBackend:
    """Sentinel registered for ``faiss``.

    FAISS is not a production backend. Running it as one would load the
    whole catalog into RAM and lose the metadata filtering and blue/green
    lifecycle that the pgvector path provides. The benchmark script
    (``scripts/bench_faiss.py``) uses FAISS directly, outside the registry.
    """

    def __init__(self) -> None:
        raise RuntimeError(
            "FAISS is not a production backend (ADR-0012). "
            "Use the benchmark script with the 'bench' extra instead: "
            "`pip install -e '.[bench]' && python scripts/bench_faiss.py`."
        )


_REGISTRY: dict[IndexBackendEnum, type[Any]] = {}


def register_backend(backend: IndexBackendEnum, backend_class: type[Any]) -> None:
    """Register ``backend_class`` for ``backend``.

    The class is stored, not instantiated. Overwriting an existing entry is
    allowed (this is the test-injection seam); callers that want to restore
    the default should capture it via :func:`get_backend` first.
    """
    _REGISTRY[backend] = backend_class
    # Debug, not info: registration happens at module import time, and a
    # module load is not an event worth surfacing at the default log level.
    # A future plugin loader that registers backends at runtime can promote
    # this; today the only caller is the module's own bottom section.
    log.debug(
        "retrieval.backend.registered",
        backend=backend.value,
        klass=backend_class.__name__,
    )


def get_backend(backend: IndexBackendEnum) -> type[Any]:
    """Return the class registered for ``backend``.

    Raises :class:`KeyError` if the enum value has no registration, which
    is a programming error (an enum value without a default).
    """
    try:
        return _REGISTRY[backend]
    except KeyError as exc:
        raise KeyError(
            f"no backend registered for {backend!r}; "
            f"registered: {sorted(k.value for k in _REGISTRY)}"
        ) from exc


def create_backend(backend: IndexBackendEnum, *args: Any, **kwargs: Any) -> IndexBackend:
    """Instantiate the class registered for ``backend``.

    Arguments are forwarded to the constructor. This helper is convenient
    for zero-argument backends (test fakes) and is the shape the M0 test
    exercised. Production callers construct classes directly, because they
    need to control the arguments:

    - :meth:`PgvectorBackend.from_registry` takes a live connection.
    - :meth:`NumpyBackend.from_run_directory` takes a run directory.
    """
    cls = get_backend(backend)
    return cls(*args, **kwargs)  # type: ignore[no-any-return]


# ============================================================================
# Default registrations.
#
# Importing recsys.retrieval.pgvector triggers the ADR-0012 lazy guard for
# psycopg; the module imports cleanly even when the 'db' extra is absent,
# because psycopg is only required at construction time.
# ============================================================================
from recsys.retrieval.pgvector import PgvectorBackend  # noqa: E402

register_backend(IndexBackendEnum.PGVECTOR, PgvectorBackend)
register_backend(IndexBackendEnum.FAISS, _FaissNotAProductionBackend)


__all__ = [
    "create_backend",
    "get_backend",
    "register_backend",
]
