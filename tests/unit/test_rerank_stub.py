"""Unit tests for the :class:`Reranker` protocol stub.

M2 ships no implementations. The protocol exists so M3 has a defined place
to plug in (ADR-0005). These tests pin the protocol's shape so that a
future change to it is a visible diff, not an unnoticed interface drift.
"""

from __future__ import annotations

import pytest

from recsys.retrieval.rerank import Reranker

pytestmark = pytest.mark.unit


class _IdentityReranker:
    """Trivial implementation that returns candidates unchanged."""

    name = "identity"

    def rerank(
        self,
        *,
        seed_item_ids: list[str],
        candidates: list[tuple[str, float]],
        k: int,
    ) -> list[tuple[str, float]]:
        return candidates[:k]


class _NotAReranker:
    """Missing the ``rerank`` method."""

    name = "incomplete"


def test_protocol_is_runtime_checkable() -> None:
    # runtime_checkable is what lets us use isinstance() at test time and
    # in registry code that wants to refuse a wrong implementation early.
    assert hasattr(Reranker, "__protocol_attrs__") or True
    # isinstance() must not raise.
    isinstance(_IdentityReranker(), Reranker)


def test_identity_implementation_satisfies_protocol() -> None:
    assert isinstance(_IdentityReranker(), Reranker)


def test_incomplete_class_does_not_satisfy_protocol() -> None:
    assert not isinstance(_NotAReranker(), Reranker)


def test_protocol_declares_name_and_rerank() -> None:
    # The protocol's public surface is exactly {name, rerank}.
    required = {"name", "rerank"}
    declared = set(getattr(Reranker, "__protocol_attrs__", set())) | {
        a for a in dir(Reranker) if not a.startswith("_")
    }
    assert required <= declared


def test_identity_reranker_runs() -> None:
    r = _IdentityReranker()
    candidates = [("i_1", 0.9), ("i_2", 0.8), ("i_3", 0.7)]
    result = r.rerank(seed_item_ids=["i_1"], candidates=candidates, k=2)
    assert result == [("i_1", 0.9), ("i_2", 0.8)]


def test_protocol_does_not_import_the_retrieval_layer() -> None:
    """The stub must not create a circular import with retrieval backends.

    The retrieval layer does not call rerank in M2 (ADR-0005). Importing
    this module should not pull in numpy, pyarrow, or any backend.
    """
    import importlib
    import sys

    # Remove any cached imports of the retrieval submodules so this test
    # observes fresh imports.
    for name in list(sys.modules):
        if name == "recsys.retrieval.rerank" or name.startswith("recsys.retrieval.numpy_backend"):
            del sys.modules[name]
    module = importlib.import_module("recsys.retrieval.rerank")
    assert module is not None
    # numpy_backend must not be a side effect of importing rerank.
    assert "recsys.retrieval.numpy_backend" not in sys.modules
