"""Unit tests for API key validation (ADR-0013).

The Argon2id hashing used by ``hash_api_key`` is slow by design (a few
milliseconds per call); the tests below hash a handful of keys, so the
suite stays fast. The plaintext path is exercised separately because it
is the development-only path the prod guard refuses.
"""

from __future__ import annotations

import pytest

from recsys.api.auth.api_key import (
    hash_api_key,
    looks_like_argon2_hash,
    validate_api_key,
)
from recsys.api.auth.principal import PrincipalKind, Scope

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# hash_api_key + looks_like_argon2_hash
# --------------------------------------------------------------------------- #
def test_argon2_hash_prefix() -> None:
    h = hash_api_key("some-key")
    assert h.startswith("$argon2")
    assert looks_like_argon2_hash(h)


def test_plaintext_does_not_look_like_argon2() -> None:
    assert not looks_like_argon2_hash("plaintext-key")


def test_hash_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        hash_api_key("")


# --------------------------------------------------------------------------- #
# validate_api_key — read path
# --------------------------------------------------------------------------- #
def test_read_key_authenticates() -> None:
    h = hash_api_key("read-key")
    p = validate_api_key(
        "read-key",
        read_entries=frozenset({h}),
        admin_entries=frozenset(),
    )
    assert p is not None
    assert p.kind == PrincipalKind.API_KEY
    assert p.has_scope(Scope.ADMIN) is False
    assert p.subject.startswith("api_key:")


def test_read_key_does_not_grant_admin() -> None:
    h = hash_api_key("read-key")
    p = validate_api_key(
        "read-key",
        read_entries=frozenset({h}),
        admin_entries=frozenset(),
    )
    assert p is not None
    assert Scope.ADMIN.value not in p.scopes


def test_unknown_key_returns_none() -> None:
    h = hash_api_key("read-key")
    assert (
        validate_api_key(
            "other-key",
            read_entries=frozenset({h}),
            admin_entries=frozenset(),
        )
        is None
    )


def test_empty_key_returns_none() -> None:
    h = hash_api_key("read-key")
    assert (
        validate_api_key(
            "",
            read_entries=frozenset({h}),
            admin_entries=frozenset(),
        )
        is None
    )


def test_no_entries_returns_none() -> None:
    assert (
        validate_api_key(
            "any",
            read_entries=frozenset(),
            admin_entries=frozenset(),
        )
        is None
    )


# --------------------------------------------------------------------------- #
# validate_api_key — admin path
# --------------------------------------------------------------------------- #
def test_admin_key_grants_admin_scope() -> None:
    h = hash_api_key("admin-key")
    p = validate_api_key(
        "admin-key",
        read_entries=frozenset(),
        admin_entries=frozenset({h}),
    )
    assert p is not None
    assert p.has_scope(Scope.ADMIN) is True


def test_admin_key_does_not_need_read_entry() -> None:
    # An admin key works even when the read list does not contain it.
    admin = hash_api_key("admin-key")
    read = hash_api_key("read-key")
    p = validate_api_key(
        "admin-key",
        read_entries=frozenset({read}),
        admin_entries=frozenset({admin}),
    )
    assert p is not None
    assert p.has_scope(Scope.ADMIN) is True


# --------------------------------------------------------------------------- #
# validate_api_key — plaintext (dev path)
# --------------------------------------------------------------------------- #
def test_plaintext_entry_matches_in_dev() -> None:
    # The dev path: an entry that is not an Argon2 hash is compared as
    # plaintext. This is what makes the development environment's
    # readable keys work; the prod guard refuses it.
    p = validate_api_key(
        "dev-key",
        read_entries=frozenset({"dev-key"}),
        admin_entries=frozenset(),
    )
    assert p is not None


def test_plaintext_entry_does_not_match_wrong_value() -> None:
    assert (
        validate_api_key(
            "wrong",
            read_entries=frozenset({"dev-key"}),
            admin_entries=frozenset(),
        )
        is None
    )


# --------------------------------------------------------------------------- #
# credential hash
# --------------------------------------------------------------------------- #
def test_credential_hash_is_stable_and_32_hex() -> None:
    h = hash_api_key("k")
    p1 = validate_api_key("k", read_entries=frozenset({h}), admin_entries=frozenset())
    p2 = validate_api_key("k", read_entries=frozenset({h}), admin_entries=frozenset())
    assert p1 is not None
    assert p2 is not None
    assert p1.credential_hash == p2.credential_hash
    assert len(p1.credential_hash) == 32


def test_different_keys_have_different_credential_hashes() -> None:
    h1 = hash_api_key("k1")
    h2 = hash_api_key("k2")
    p1 = validate_api_key("k1", read_entries=frozenset({h1}), admin_entries=frozenset())
    p2 = validate_api_key("k2", read_entries=frozenset({h2}), admin_entries=frozenset())
    assert p1 is not None
    assert p2 is not None
    assert p1.credential_hash != p2.credential_hash
