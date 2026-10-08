"""Unit tests for JWT validation (ADR-0013).

Every test pins the clock by passing ``now`` to ``validate_jwt``; the
function is pure for a fixed clock, which is what makes these tests
fast and deterministic.
"""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest

from recsys.api.auth.jwt import JwtError, validate_jwt
from recsys.api.auth.principal import PrincipalKind

pytestmark = pytest.mark.unit

_SECRET = "x" * 40
_ALGO = "HS256"


def _token(
    *,
    sub: str = "u_1",
    scope: str | None = None,
    exp_offset: int = 60,
    iat_offset: int = 0,
    secret: str = _SECRET,
    algorithm: str = _ALGO,
    extra: dict[str, object] | None = None,
) -> str:
    now = int(time.time())
    payload: dict[str, object] = {
        "sub": sub,
        "exp": now + exp_offset,
        "iat": now + iat_offset,
    }
    if scope is not None:
        payload["scope"] = scope
    if extra:
        payload.update(extra)
    return pyjwt.encode(payload, secret, algorithm=algorithm)


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_valid_token() -> None:
    now = int(time.time())
    token = _token()
    p = validate_jwt(token, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600, now=now)
    assert p.subject == "u_1"
    assert p.kind == PrincipalKind.JWT
    assert p.expires_at is not None
    assert p.expires_at > now


def test_token_with_admin_scope() -> None:
    token = _token(scope="admin")
    p = validate_jwt(
        token,
        secret=_SECRET,
        algorithm=_ALGO,
        max_age_seconds=3600,
        now=int(time.time()),
    )
    assert "admin" in p.scopes


def test_token_with_multi_word_scope() -> None:
    token = _token(scope="admin write")
    p = validate_jwt(
        token,
        secret=_SECRET,
        algorithm=_ALGO,
        max_age_seconds=3600,
        now=int(time.time()),
    )
    assert p.scopes == frozenset({"admin", "write"})


def test_token_without_scope_claim() -> None:
    token = _token()
    p = validate_jwt(
        token,
        secret=_SECRET,
        algorithm=_ALGO,
        max_age_seconds=3600,
        now=int(time.time()),
    )
    assert p.scopes == frozenset()


def test_token_with_empty_scope_string() -> None:
    token = _token(scope="")
    p = validate_jwt(
        token,
        secret=_SECRET,
        algorithm=_ALGO,
        max_age_seconds=3600,
        now=int(time.time()),
    )
    assert p.scopes == frozenset()


# --------------------------------------------------------------------------- #
# missing credential
# --------------------------------------------------------------------------- #
def test_empty_token_raises_missing() -> None:
    with pytest.raises(JwtError) as exc:
        validate_jwt("", secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600)
    assert exc.value.reason == "missing"


# --------------------------------------------------------------------------- #
# signature / structure
# --------------------------------------------------------------------------- #
def test_wrong_secret_raises_invalid() -> None:
    token = _token()
    with pytest.raises(JwtError) as exc:
        validate_jwt(
            token,
            secret="y" * 40,
            algorithm=_ALGO,
            max_age_seconds=3600,
        )
    assert exc.value.reason == "invalid"


def test_malformed_token_raises_invalid() -> None:
    with pytest.raises(JwtError) as exc:
        validate_jwt(
            "not.a.jwt",
            secret=_SECRET,
            algorithm=_ALGO,
            max_age_seconds=3600,
        )
    assert exc.value.reason == "invalid"


def test_missing_sub_raises_invalid() -> None:
    now = int(time.time())
    payload = {"exp": now + 60, "iat": now}
    token = pyjwt.encode(payload, _SECRET, algorithm=_ALGO)
    with pytest.raises(JwtError) as exc:
        validate_jwt(token, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600)
    assert exc.value.reason == "invalid"


def test_empty_sub_raises_invalid() -> None:
    token = _token(sub="")
    with pytest.raises(JwtError) as exc:
        validate_jwt(token, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600)
    assert exc.value.reason == "invalid"


def test_missing_iat_raises_invalid() -> None:
    now = int(time.time())
    payload = {"sub": "u_1", "exp": now + 60}
    token = pyjwt.encode(payload, _SECRET, algorithm=_ALGO)
    with pytest.raises(JwtError) as exc:
        validate_jwt(token, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600)
    assert exc.value.reason == "invalid"


def test_missing_exp_raises_invalid() -> None:
    now = int(time.time())
    payload = {"sub": "u_1", "iat": now}
    token = pyjwt.encode(payload, _SECRET, algorithm=_ALGO)
    with pytest.raises(JwtError) as exc:
        validate_jwt(token, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600)
    assert exc.value.reason == "invalid"


def test_wrong_algorithm_raises_invalid() -> None:
    # Token signed with HS512 but the validator only accepts HS256.
    # HS512 wants a key of at least 64 bytes; use a longer one for the
    # signing step so PyJWT does not emit an InsecureKeyLengthWarning.
    now = int(time.time())
    payload = {"sub": "u_1", "exp": now + 60, "iat": now}
    hs512_secret = "z" * 64
    token = pyjwt.encode(payload, hs512_secret, algorithm="HS512")
    with pytest.raises(JwtError) as exc:
        validate_jwt(token, secret=_SECRET, algorithm="HS256", max_age_seconds=3600)
    assert exc.value.reason == "invalid"


# --------------------------------------------------------------------------- #
# expiry / max age
# --------------------------------------------------------------------------- #
def test_expired_token_raises_expired() -> None:
    token = _token(exp_offset=-10)
    with pytest.raises(JwtError) as exc:
        validate_jwt(
            token,
            secret=_SECRET,
            algorithm=_ALGO,
            max_age_seconds=7200,
        )
    assert exc.value.reason == "expired"


def test_iat_too_old_raises_iat_too_old() -> None:
    now = int(time.time())
    token = _token(exp_offset=7200, iat_offset=-7200)
    with pytest.raises(JwtError) as exc:
        validate_jwt(
            token,
            secret=_SECRET,
            algorithm=_ALGO,
            max_age_seconds=3600,
            now=now,
        )
    assert exc.value.reason == "iat_too_old"


def test_iat_within_max_age_ok() -> None:
    now = int(time.time())
    token = _token(exp_offset=7200, iat_offset=-1800)
    p = validate_jwt(
        token,
        secret=_SECRET,
        algorithm=_ALGO,
        max_age_seconds=3600,
        now=now,
    )
    assert p.subject == "u_1"


# --------------------------------------------------------------------------- #
# credential hash
# --------------------------------------------------------------------------- #
def test_credential_hash_is_stable() -> None:
    token = _token()
    now = int(time.time())
    p1 = validate_jwt(token, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600, now=now)
    p2 = validate_jwt(token, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600, now=now)
    assert p1.credential_hash == p2.credential_hash
    assert len(p1.credential_hash) == 32


def test_different_tokens_have_different_hashes() -> None:
    now = int(time.time())
    t1 = _token(sub="u_1")
    t2 = _token(sub="u_2")
    p1 = validate_jwt(t1, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600, now=now)
    p2 = validate_jwt(t2, secret=_SECRET, algorithm=_ALGO, max_age_seconds=3600, now=now)
    assert p1.credential_hash != p2.credential_hash
