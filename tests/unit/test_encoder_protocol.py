"""Unit tests for the Encoder protocol surface.

These tests do not import onnxruntime or sentence-transformers. They use a
:class:`FakeEncoder` that satisfies the protocol and verify that
implementations are interchangeable at the type level. Concrete encoders
are exercised in ``tests/integration/test_encoder_parity.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from recsys.embeddings.encoder import Encoder


class FakeEncoder:
    """Deterministic synthetic encoder for unit tests."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.model_version = "fake+v1"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows: list[list[float]] = []
        for text in texts:
            # Deterministic pseudo-random from the text bytes.
            seed = sum(text.encode("utf-8")) or 1
            rng = np.random.default_rng(seed)
            row = rng.standard_normal(self._dim).astype(np.float32)
            rows.append(row.tolist())
        arr = np.asarray(rows, dtype=np.float32)
        # L2 normalize along axis 1.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        result = (arr / np.maximum(norms, 1e-12)).astype(np.float32)
        return cast(NDArray[np.float32], result)


def test_fake_encoder_satisfies_protocol() -> None:
    enc = FakeEncoder()
    assert isinstance(enc, Encoder)


def test_protocol_members_are_reachable() -> None:
    enc: Encoder = FakeEncoder(dim=4)
    assert enc.embedding_dim == 4
    assert isinstance(enc.model_version, str)


def test_encode_returns_l2_normalized_rows() -> None:
    enc = FakeEncoder(dim=16)
    out = enc.encode(["hello", "world"])
    assert out.dtype == np.float32
    assert out.shape == (2, 16)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


def test_encode_empty_batch() -> None:
    enc = FakeEncoder(dim=4)
    out = enc.encode([])
    assert out.shape == (0, 4)
    assert out.dtype == np.float32


def test_encode_is_deterministic() -> None:
    enc = FakeEncoder(dim=8)
    a = enc.encode(["alpha", "beta", "gamma"])
    b = enc.encode(["alpha", "beta", "gamma"])
    np.testing.assert_array_equal(a, b)
