"""Tests for the retrieval backend registry."""

from __future__ import annotations

import pytest

from recsys.config.enums import IndexBackend as IndexBackendEnum
from recsys.retrieval import base, registry


class _FakeBackend:
    name = "fake"

    def is_ready(self) -> bool:
        return True

    async def search(
        self,
        *,
        vector: list[float],
        k: int,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        return [("i_1", 0.9)][:k]


def test_get_backend_returns_factory_for_registered_backend() -> None:
    factory = registry.get_backend(IndexBackendEnum.PGVECTOR)
    assert callable(factory)


def test_create_backend_delegates_to_registered_factory() -> None:
    original = registry.get_backend(IndexBackendEnum.PGVECTOR)
    try:
        registry.register_backend(IndexBackendEnum.PGVECTOR, _FakeBackend)
        backend = registry.create_backend(IndexBackendEnum.PGVECTOR)
        assert isinstance(backend, _FakeBackend)
        assert isinstance(backend, base.IndexBackend)
    finally:
        registry.register_backend(IndexBackendEnum.PGVECTOR, original)


def test_faiss_factory_raises_helpful_error_when_not_installed() -> None:
    # If faiss is installed in this environment, this test is a no-op.
    from recsys._optional import optional_import

    if optional_import("faiss") is not None:
        pytest.skip("faiss is installed; absence path cannot be exercised")

    factory = registry.get_backend(IndexBackendEnum.FAISS)
    with pytest.raises(RuntimeError, match="bench"):
        factory()
