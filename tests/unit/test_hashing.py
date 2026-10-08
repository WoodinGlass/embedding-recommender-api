"""Unit tests for the hashing helpers.

Two functions, two threat models. The BLAKE2b test pins the length and
the determinism; the HMAC test pins the key-required behavior and the
fact that a different key produces a different hash for the same input.
"""

from __future__ import annotations

import pytest

from recsys.security.hashing import blake2b_16, hmac_sha256

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# blake2b_16
# --------------------------------------------------------------------------- #
def test_blake2b_16_length() -> None:
    # 16 bytes -> 32 hex characters.
    assert len(blake2b_16("hello")) == 32


def test_blake2b_16_lowercase_hex() -> None:
    out = blake2b_16("hello")
    assert all(c in "0123456789abcdef" for c in out)


def test_blake2b_16_deterministic() -> None:
    assert blake2b_16("hello") == blake2b_16("hello")


def test_blake2b_16_distinct_inputs_distinct_hashes() -> None:
    assert blake2b_16("a") != blake2b_16("b")


def test_blake2b_16_unicode_input() -> None:
    # A non-ASCII input hashes deterministically too.
    assert blake2b_16("café") == blake2b_16("café")


def test_blake2b_16_known_value() -> None:
    # Pinned: a change to the algorithm or the digest size fails this.
    # The value was computed with hashlib.blake2b(b"hello", digest_size=16).
    import hashlib

    expected = hashlib.blake2b(b"hello", digest_size=16).hexdigest()
    assert blake2b_16("hello") == expected


# --------------------------------------------------------------------------- #
# hmac_sha256
# --------------------------------------------------------------------------- #
def test_hmac_sha256_length() -> None:
    # SHA256 output is 32 bytes -> 64 hex characters.
    assert len(hmac_sha256("key", "message")) == 64


def test_hmac_sha256_deterministic() -> None:
    assert hmac_sha256("key", "msg") == hmac_sha256("key", "msg")


def test_hmac_sha256_key_sensitivity() -> None:
    # Two different keys over the same message produce different hashes.
    assert hmac_sha256("key1", "msg") != hmac_sha256("key2", "msg")


def test_hmac_sha256_message_sensitivity() -> None:
    assert hmac_sha256("key", "msg1") != hmac_sha256("key", "msg2")


def test_hmac_sha256_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="non-empty key"):
        hmac_sha256("", "msg")


def test_hmac_sha256_accepts_bytes_and_str_equivalently() -> None:
    assert hmac_sha256("k", "m") == hmac_sha256(b"k", b"m")


def test_hmac_sha256_known_value() -> None:
    # Pinned: a change to the algorithm fails this.
    import hashlib
    import hmac as _hmac

    expected = _hmac.new(b"key", b"message", hashlib.sha256).hexdigest()
    assert hmac_sha256("key", "message") == expected
