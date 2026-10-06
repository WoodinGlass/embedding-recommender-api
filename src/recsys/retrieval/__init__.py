"""Retrieval backends (pgvector, FAISS) and their registry."""

from recsys.retrieval.base import IndexBackend
from recsys.retrieval.registry import (
    create_backend,
    get_backend,
    register_backend,
)

__all__ = ["IndexBackend", "create_backend", "get_backend", "register_backend"]
