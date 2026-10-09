"""Unit tests for user id hashing (ADR-0018)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from recsys.events import UserIdHash, hash_user_id


# ---------------------------------------------------------------- #
# hash_user_id
# ---------------------------------------------------------------- #
def test_returns_hash_and_version() -> None:
    h = hash_user_id("u_123", salt="s", salt_version=1)
    assert isinstance(h, UserIdHash)
    assert h.version == 1
    assert len(h.hex) == 64
    assert all(c in "0123456789abcdef" for c in h.hex)


def test_deterministic() -> None:
    a = hash_user_id("u_123", salt="s", salt_version=1)
    b = hash_user_id("u_123", salt="s", salt_version=1)
    assert a == b


def test_different_user_produces_different_hash() -> None:
    a = hash_user_id("u_123", salt="s", salt_version=1)
    b = hash_user_id("u_456", salt="s", salt_version=1)
    assert a.hex != b.hex


def test_different_salt_produces_different_hash() -> None:
    a = hash_user_id("u_123", salt="s1", salt_version=1)
    b = hash_user_id("u_123", salt="s2", salt_version=1)
    assert a.hex != b.hex


def test_salt_version_does_not_change_hash() -> None:
    """Rotating the version without rotating the salt produces the
    same hex. The version is metadata, not a second salt; a real
    rotation changes both, and this test pins that the version is
    not silently mixed in (which would surprise a caller who bumps
    it for bookkeeping)."""
    a = hash_user_id("u_123", salt="s", salt_version=1)
    b = hash_user_id("u_123", salt="s", salt_version=2)
    assert a.hex == b.hex
    assert a.version != b.version


def test_does_not_leak_raw_user_id() -> None:
    h = hash_user_id("user@example.com", salt="s", salt_version=1)
    assert "user@example.com" not in h.hex


def test_hash_is_stable_across_calls_with_same_inputs() -> None:
    """A stable hash is what makes grouping events by user possible
    (ADR-0018 § Alternatives: a non-reproducible hash cannot group)."""
    values = {hash_user_id("u_1", salt="s", salt_version=0).hex for _ in range(50)}
    assert len(values) == 1


# ---------------------------------------------------------------- #
# guards
# ---------------------------------------------------------------- #
def test_empty_user_id_raises() -> None:
    with pytest.raises(ValueError, match="user_id"):
        hash_user_id("", salt="s", salt_version=0)


def test_empty_salt_raises() -> None:
    with pytest.raises(ValueError, match="salt"):
        hash_user_id("u_1", salt="", salt_version=0)


def test_negative_salt_version_raises() -> None:
    with pytest.raises(ValueError, match="salt_version"):
        hash_user_id("u_1", salt="s", salt_version=-1)


# ---------------------------------------------------------------- #
# UserIdHash dataclass
# ---------------------------------------------------------------- #
def test_user_id_hash_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        UserIdHash(hex="abc", version=0)


def test_user_id_hash_rejects_uppercase_hex() -> None:
    with pytest.raises(ValueError, match="64 lowercase hex"):
        UserIdHash(hex="A" * 64, version=0)


def test_user_id_hash_rejects_negative_version() -> None:
    with pytest.raises(ValueError, match="version"):
        UserIdHash(hex="a" * 64, version=-1)


def test_user_id_hash_is_frozen() -> None:
    h = UserIdHash(hex="a" * 64, version=1)
    with pytest.raises(FrozenInstanceError):
        h.hex = "b" * 64  # type: ignore[misc]
