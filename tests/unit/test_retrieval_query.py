"""Unit tests for query vector resolution (ADR-0009 § 3).

A fake connection and a fake encoder; the integration test in
``tests/integration/test_retrieval_query.py`` exercises the real
database round trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.retrieval.query import (
    SEED_TEXT_SEPARATOR,
    resolve_query_vector,
)


# ---------------------------------------------------------------- #
# fakes
# ---------------------------------------------------------------- #
class _Cursor:
    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self._rows = rows
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple[str, str, str]]:
        return list(self._rows)

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


class _Conn:
    def __init__(self, rows: list[tuple[str, str, str]]) -> None:
        self._rows = rows

    def cursor(self) -> _Cursor:
        return _Cursor(self._rows)


class _Encoder:
    """Returns a fixed vector per text; encodes the count and the
    text passed in so a test can assert the payload."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.seen: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.seen.append(list(texts))
        n = len(texts)
        # Deterministic but direction-rich so the mean is not zero.
        base = np.array([1.0, 2.0, 3.0, 4.0][: self._dim], dtype=np.float32)
        return np.tile(base, (n, 1)).astype(np.float32)


# ---------------------------------------------------------------- #
# happy path
# ---------------------------------------------------------------- #
def test_returns_normalized_mean() -> None:
    conn = _Conn([("i_1", "t1", "d1"), ("i_2", "t2", "d2")])
    enc = _Encoder()
    vec = resolve_query_vector(conn, encoder=enc, seed_item_ids=["i_1", "i_2"])
    assert vec is not None
    assert vec.dtype == np.float32
    assert vec.shape == (4,)
    # The mean of the identical rows is the row; L2-normalized.
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6


def test_uses_title_description_with_separator() -> None:
    conn = _Conn([("i_1", "Title A", "Desc A")])
    enc = _Encoder()
    resolve_query_vector(conn, encoder=enc, seed_item_ids=["i_1"])
    assert enc.seen == [[f"Title A{SEED_TEXT_SEPARATOR}Desc A"]]


def test_returns_none_for_empty_seed_list() -> None:
    conn = _Conn([])
    enc = _Encoder()
    assert resolve_query_vector(conn, encoder=enc, seed_item_ids=[]) is None
    # No query issued; the connection was never touched.
    # (The fake does not track connection.cursor calls, but the
    # encoder's `seen` proves the encoder was not called.)
    assert enc.seen == []


def test_returns_none_when_no_rows_match() -> None:
    conn = _Conn([])
    enc = _Encoder()
    vec = resolve_query_vector(conn, encoder=enc, seed_item_ids=["missing"])
    assert vec is None
    assert enc.seen == []


def test_sorts_by_item_id_for_stable_order() -> None:
    conn = _Conn([("i_2", "t2", "d2"), ("i_1", "t1", "d1")])
    enc = _Encoder()
    resolve_query_vector(conn, encoder=enc, seed_item_ids=["i_2", "i_1"])
    # Sorted by item_id; not the request order.
    assert enc.seen == [[f"t1{SEED_TEXT_SEPARATOR}d1", f"t2{SEED_TEXT_SEPARATOR}d2"]]


def test_zero_mean_returns_none() -> None:
    """A symmetric seed set can produce a zero mean; the function
    returns None rather than normalize the zero vector."""

    class _ZeroEncoder:
        def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
            # +1 in one axis, -1 in another, equal counts -> mean 0.
            n = len(texts)
            arr = np.zeros((n, 2), dtype=np.float32)
            arr[: n // 2, 0] = 1.0
            arr[n // 2 :, 0] = -1.0
            return arr

    conn = _Conn([("i_1", "t", "d"), ("i_2", "t", "d")])
    assert resolve_query_vector(conn, encoder=_ZeroEncoder(), seed_item_ids=["i_1", "i_2"]) is None


def test_shape_mismatch_raises() -> None:
    class _BadEncoder:
        def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
            return np.zeros((1, 4), dtype=np.float32)  # wrong row count

    conn = _Conn([("i_1", "t", "d"), ("i_2", "t", "d")])
    with pytest.raises(RuntimeError, match="rows"):
        resolve_query_vector(conn, encoder=_BadEncoder(), seed_item_ids=["i_1", "i_2"])
