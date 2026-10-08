"""API key validation (ADR-0013).

API keys are stored as Argon2id hashes. A configuration entry that
starts with ``$argon2`` is a hash; anything else is treated as plaintext
and is rejected in production by ``Settings._prod_guard``. The plaintext
path exists so the development environment can have readable keys
without a hashing step; it is not a supported production shape.

Argon2id verification costs a few milliseconds per entry. A deployment
with several keys pays that cost per entry on every request. The
mitigation (an LRU cache of recently verified keys, keyed on the
credential hash) is documented in ADR-0013 and is not implemented
until a measurement shows it is needed.
"""

from __future__ import annotations

import secrets

from passlib.context import CryptContext

from recsys.api.auth.principal import Principal, PrincipalKind, Scope
from recsys.security.hashing import blake2b_16

#: The prefix that marks an entry as an Argon2id hash. Argon2 hashes in
#: PHC format start with ``$argon2id$``, ``$argon2i$``, or ``$argon2d$``;
#: the common prefix is enough to route without parsing.
_ARGON2_PREFIX = "$argon2"

_CTX = CryptContext(schemes=["argon2"], deprecated="auto")


def looks_like_argon2_hash(value: str) -> bool:
    """Return True when ``value`` is an Argon2id PHC hash, not plaintext."""
    return value.startswith(_ARGON2_PREFIX)


def hash_api_key(plaintext: str) -> str:
    """Return the Argon2id hash of ``plaintext``.

    Used by the operator tooling (a Makefile target and the
    ``docs/ops.md`` section 8.1 rotation procedure); the API never calls
    it.
    """
    if not plaintext:
        raise ValueError("cannot hash an empty API key")
    hashed: str = _CTX.hash(plaintext)
    return hashed


def validate_api_key(
    plaintext: str,
    *,
    read_entries: frozenset[str],
    admin_entries: frozenset[str],
) -> Principal | None:
    """Return a :class:`Principal` for ``plaintext``, or ``None``.

    A key that matches an entry in ``admin_entries`` receives the
    ``admin`` scope in addition to being authenticated. A key that
    matches only ``read_entries`` is authenticated without any scope.
    A key that matches nothing is ``None``.

    An empty ``plaintext`` returns ``None`` without touching the
    configured entries, so an empty header is not a verification (and
    not a timing signal against a stored hash).
    """
    if not plaintext:
        return None

    has_admin = any(_verify_one(plaintext, entry) for entry in admin_entries)
    if has_admin:
        scopes = frozenset({Scope.ADMIN.value})
    else:
        has_read = any(_verify_one(plaintext, entry) for entry in read_entries)
        if not has_read:
            return None
        scopes = frozenset()

    credential_hash = blake2b_16(plaintext)
    return Principal(
        subject=f"api_key:{credential_hash[:8]}",
        scopes=scopes,
        kind=PrincipalKind.API_KEY,
        credential_hash=credential_hash,
    )


def _verify_one(plaintext: str, entry: str) -> bool:
    """Return True when ``plaintext`` matches ``entry``.

    An Argon2id hash is verified with passlib (constant time with
    respect to the hash content). A plaintext entry is compared with
    :func:`secrets.compare_digest`; the comparison is constant-time for
    equal-length inputs and short-circuits for unequal ones, which is
    fine for the development-only path this serves.
    """
    if looks_like_argon2_hash(entry):
        try:
            return bool(_CTX.verify(plaintext, entry))
        except Exception:
            # A malformed hash in the config should not crash every
            # request. The prod guard refuses plaintext entries, not
            # malformed hashes; a review of the config is the fix.
            return False
    return secrets.compare_digest(entry, plaintext)
