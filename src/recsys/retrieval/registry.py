"""Registry pattern for retrieval backends.

Adding a backend is a matter of registering a factory under a new
:class:`IndexBackend` value — there is no ``if/elif`` chain to edit here or at
the call site. Tests use :func:`register_backend` to inject fakes without
touching production code.
"""

from __future__ import annotations

from collections.abc import Callable

from recsys._optional import optional_import
from recsys.config.enums import IndexBackend as IndexBackendEnum
from recsys.monitoring.logging import get_logger
from recsys.retrieval.base import IndexBackend

log = get_logger(__name__)

BackendFactory = Callable[[], IndexBackend]


def _pgvector_factory() -> IndexBackend:
    raise NotImplementedError(
        "pgvector backend lands in M2 (Retrieval and offline evaluation)."
    )


def _faiss_factory() -> IndexBackend:
    faiss = optional_import("faiss")
    if faiss is None:
        raise RuntimeError(
            "INDEX_BACKEND=faiss requires the 'bench' extra. "
            "Install with: pip install -e '.[bench]'"
        )
    raise NotImplementedError(
        "FAISS backend lands in M2 (Retrieval and offline evaluation)."
    )


_REGISTRY: dict[IndexBackendEnum, BackendFactory] = {
    IndexBackendEnum.PGVECTOR: _pgvector_factory,
    IndexBackendEnum.FAISS: _faiss_factory,
}


def register_backend(
    backend: IndexBackendEnum, factory: BackendFactory
) -> None:
    """Register or override a backend factory."""
    _REGISTRY[backend] = factory


def get_backend(backend: IndexBackendEnum) -> BackendFactory:
    """Return the factory for *backend*.

    Raises :class:`KeyError` if the enum has no registered factory, which
    would indicate a programming error (an enum value without an entry).
    """
    try:
        return _REGISTRY[backend]
    except KeyError as exc:
        raise KeyError(f"No factory registered for backend {backend!r}") from exc


def create_backend(backend: IndexBackendEnum) -> IndexBackend:
    """Instantiate the backend for *backend*."""
    factory = get_backend(backend)
    log.info("retrieval.backend.create", backend=backend.value)
    return factory()
