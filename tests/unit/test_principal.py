"""Unit tests for the Principal dataclass and Scope enum."""

from __future__ import annotations

import pytest

from recsys.api.auth.principal import Principal, PrincipalKind, Scope

pytestmark = pytest.mark.unit


def _principal(
    *,
    scopes: frozenset[str] | None = None,
    credential_hash: str = "0" * 32,
) -> Principal:
    return Principal(
        subject="u_1",
        scopes=scopes if scopes is not None else frozenset(),
        kind=PrincipalKind.API_KEY,
        credential_hash=credential_hash,
    )


def test_scope_enum_values() -> None:
    assert Scope.ADMIN.value == "admin"


def test_principal_kind_enum_values() -> None:
    assert PrincipalKind.API_KEY.value == "api_key"
    assert PrincipalKind.JWT.value == "jwt"


def test_has_scope_admin() -> None:
    p = _principal(scopes=frozenset({Scope.ADMIN.value}))
    assert p.has_scope(Scope.ADMIN) is True


def test_has_scope_missing() -> None:
    p = _principal()
    assert p.has_scope(Scope.ADMIN) is False


def test_credential_hash_prefix_is_8_chars() -> None:
    p = _principal(credential_hash="abcdef0123456789" + "0" * 16)
    assert p.credential_hash_prefix == "abcdef01"
    assert len(p.credential_hash_prefix) == 8


def test_principal_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    p = _principal()
    with pytest.raises(FrozenInstanceError):
        p.subject = "other"  # type: ignore[misc]


def test_principal_default_expires_at_is_none() -> None:
    p = _principal()
    assert p.expires_at is None


def test_principal_with_expires_at() -> None:
    p = Principal(
        subject="u",
        scopes=frozenset(),
        kind=PrincipalKind.JWT,
        credential_hash="0" * 32,
        expires_at=1_700_000_000,
    )
    assert p.expires_at == 1_700_000_000
