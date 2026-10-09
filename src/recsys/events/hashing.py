"""User id hashing for event ingestion (ADR-0018 § User id hashing).

The client sends a raw ``user_id``; the server stores an HMAC-SHA256
of it under a versioned secret salt. The raw id never reaches
storage, logs, traces, or metric labels; only the hash does.

Why HMAC and not SHA256: a plain SHA256 of a ``user_id`` is
reversible against a dictionary when the id space is small or
predictable (an email address, a phone number, a sequential id).
HMAC with a secret key removes that attack — an attacker who has the
hashes but not the salt cannot enumerate the input space.

Why a salt version: rotating the salt produces different hashes for
the same ``user_id``. Without a version marker the analysis would
see one user as two. The version is stored alongside every hash so
a rotation is a filter, not a data loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from recsys.security.hashing import hmac_sha256


@dataclass(frozen=True)
class UserIdHash:
    """The hash of a user id, plus the salt version that produced it.

    Both fields are stored on every event that carries a user
    (ADR-0018). ``hex`` is a 64-character lowercase hex string
    (HMAC-SHA256); ``version`` is an integer the operator bumps on
    rotation.
    """

    hex: str
    version: int

    def __post_init__(self) -> None:
        if len(self.hex) != 64 or not all(c in "0123456789abcdef" for c in self.hex):
            raise ValueError("UserIdHash.hex must be 64 lowercase hex characters")
        if self.version < 0:
            raise ValueError(f"UserIdHash.version must be >= 0, got {self.version}")


def hash_user_id(
    user_id: str,
    *,
    salt: str,
    salt_version: int,
) -> UserIdHash:
    """Return the HMAC-SHA256 of ``user_id`` under ``salt``.

    The caller passes the same ``salt`` and ``salt_version`` the
    process is configured with (``USER_ID_HASH_SALT`` and
    ``USER_ID_HASH_SALT_VERSION``). The salt is required and must
    not be empty: an empty key weakens HMAC to an unkeyed hash,
    which is the case the versioned-salt design exists to avoid.

    ``user_id`` must be a non-empty string. The caller is
    responsible for not passing an already-hashed value
    (double-hashing is stable but produces a different identifier;
    ADR-0018 § Consequences notes the convention).
    """
    if not user_id:
        raise ValueError("user_id must be a non-empty string")
    if not salt:
        raise ValueError("salt must be a non-empty string")
    if salt_version < 0:
        raise ValueError(f"salt_version must be >= 0, got {salt_version}")

    digest = hmac_sha256(salt, user_id)
    return UserIdHash(hex=digest, version=salt_version)


__all__ = [
    "UserIdHash",
    "hash_user_id",
]
