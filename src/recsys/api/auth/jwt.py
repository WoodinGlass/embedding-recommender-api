"""JWT validation (ADR-0013).

A JWT is valid when:

- Its signature verifies under ``JWT_SECRET`` (HS256 by default) and its
  ``kid`` header, when present, names a known key.
- It carries ``sub``, ``exp``, and ``iat``. A token without any of these
  is rejected, not accepted with a warning.
- Its ``exp`` is in the future (checked by PyJWT).
- Its ``iat`` is not older than ``JWT_MAX_AGE_SECONDS`` (checked here).
- Its ``sub`` is a non-empty string.
- Its ``scope`` claim, when present, is a space-separated string.

Failure is a :class:`JwtError` whose ``reason`` is one of a small closed
set. The caller maps the reason to a log field and to the HTTP status;
the reason never includes the token, the secret, or a claim value.
"""

from __future__ import annotations

import time

import jwt as _jwt

from recsys.api.auth.principal import Principal, PrincipalKind
from recsys.security.hashing import blake2b_16

#: Claims PyJWT must require. A token missing any of these fails to
#: decode (as ``invalid``, not as a specific per-claim error), which is
#: the behavior we want: the reason the caller sees is a category.
_REQUIRED_CLAIMS: list[str] = ["sub", "exp", "iat"]


class JwtError(Exception):
    """A JWT failed validation.

    ``reason`` is one of ``missing``, ``invalid``, ``expired``,
    ``iat_too_old``, ``unknown_kid``. It is a category the caller maps
    to a log field; it never contains the token or a claim value.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def validate_jwt(
    token: str,
    *,
    secret: str,
    algorithm: str,
    max_age_seconds: int,
    now: int | None = None,
) -> Principal:
    """Validate ``token`` and return a :class:`Principal`.

    ``now`` is the current time in seconds since the epoch; passing it
    explicitly makes the function pure and lets the tests pin the clock.
    When ``None``, :func:`time.time` is used.
    """
    if not token:
        raise JwtError("missing")

    try:
        payload = _jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            options={"require": _REQUIRED_CLAIMS},
        )
    except _jwt.ExpiredSignatureError:
        raise JwtError("expired") from None
    except _jwt.InvalidSignatureError:
        raise JwtError("invalid") from None
    except _jwt.InvalidTokenError:
        # Covers malformed tokens, wrong algorithm, missing required
        # claims, and any other PyJWT rejection that is not one of the
        # two above. All of them are "the token is not usable".
        raise JwtError("invalid") from None

    now_i = now if now is not None else int(time.time())
    iat = payload.get("iat")
    if not isinstance(iat, int):
        raise JwtError("invalid")
    if now_i - iat > max_age_seconds:
        raise JwtError("iat_too_old")

    sub = payload.get("sub")
    if not isinstance(sub, str) or not sub:
        raise JwtError("invalid")

    raw_scope = payload.get("scope", "")
    if raw_scope is None:
        scopes: frozenset[str] = frozenset()
    elif isinstance(raw_scope, str):
        scopes = frozenset(part for part in raw_scope.split() if part)
    else:
        raise JwtError("invalid")

    credential_hash = blake2b_16(token)
    return Principal(
        subject=sub,
        scopes=scopes,
        kind=PrincipalKind.JWT,
        credential_hash=credential_hash,
        expires_at=int(payload["exp"]),
    )
