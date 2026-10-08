"""Unit tests for ``_coerce_vector`` in the pgvector backend.

The helper exists because the driver's representation of a ``vector``
column depends on whether the pgvector adapter is registered. The
evaluation script does not register it (the connection is shared), so
the column comes back as text; ``_coerce_vector`` handles text, bytes,
ndarray, and list alike. Catching the real string form here means the
integration job is not the first place a parse bug surfaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.retrieval.pgvector import _coerce_vector

pytestmark = pytest.mark.unit


def test_ndarray_passthrough() -> None:
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = _coerce_vector(v)
    assert out.dtype == np.float32
    assert np.array_equal(out, v)


def test_string_bracketed() -> None:
    out = _coerce_vector("[0.1,0.2,0.3]")
    assert out.dtype == np.float32
    assert out.shape == (3,)
    assert out[0] == pytest.approx(0.1)
    assert out[2] == pytest.approx(0.3)


def test_string_with_spaces_and_newlines() -> None:
    """The CI error showed pgvector returning text with scientific
    notation and no spaces around the brackets. The parser must be
    indifferent to whitespace."""
    raw = "[0.0054489584, 0.03685521, -0.013285993, 3.0158315e-05]"
    out = _coerce_vector(raw)
    assert out.shape == (4,)
    assert out[0] == pytest.approx(0.0054489584)
    assert out[3] == pytest.approx(3.0158315e-05)


def test_string_empty_brackets() -> None:
    out = _coerce_vector("[]")
    assert out.shape == (0,)


def test_bytes_decoded_then_parsed() -> None:
    out = _coerce_vector(b"[1.0,2.0]")
    assert out.shape == (2,)
    assert out[0] == pytest.approx(1.0)


def test_python_list() -> None:
    out = _coerce_vector([0.5, -0.5])
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.5)


def test_malformed_raises_value_error() -> None:
    # float() raises ``ValueError: could not convert string to float: ...``
    with pytest.raises(ValueError, match="could not convert"):
        _coerce_vector("[1.0, not_a_number]")


def test_realistic_pgvector_string_parses() -> None:
    """A string with the exact shape the CI error carried: no spaces, a
    trailing scientific notation value, and 384 elements. This is a
    regression guard: the CI bug was exactly this shape."""
    values = [str(0.001 * i) for i in range(383)] + ["3.0158315e-05"]
    raw = "[" + ",".join(values) + "]"
    out = _coerce_vector(raw)
    assert out.shape == (384,)
    assert out.dtype == np.float32
