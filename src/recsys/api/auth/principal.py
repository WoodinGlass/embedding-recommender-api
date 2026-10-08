"""The :class:`Principal`: what a validated credential becomes.

A credential (an API key or a JWT) is validated exactly once per request
and never used again. What flows through the request is a ``Principal``
— the subject, the scopes, and a hash of the credential for keying the
rate limiter and the logs. The raw credential is discarded at the
validation boundary (ADR-0013, "No credential in logs, metrics, or
errors").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PrincipalKind(StrEnum):
    """Which credential model produced this Principal."""

    API_KEY = "api_key"
    JWT = "jwt"


class Scope(StrEnum):
    """Scopes a Principal can hold.

    Being authenticated is implicit and is not a scope. Only scopes that
    are *additional* to being authenticated are enumerated here. A
    future scope is a new member; adding one is a contract change
    (``docs/contracts.md`` section 2.5).
    """

    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    """A validated caller.

    ``subject`` is opaque: for an API key it is derived from the key's
    hash prefix; for a JWT it is the token's ``sub`` claim. Downstream
    code must not parse it.

    ``credential_hash`` is the BLAKE2b-128 hash of the presented
    credential. It is the key the rate limiter uses (ADR-0014) and the
    value the access log records. It never reveals the credential.

    ``expires_at`` is the JWT ``exp`` claim in seconds since the epoch,
    or ``None`` for an API key (which does not expire).
    """

    subject: str
    scopes: frozenset[str]
    kind: PrincipalKind
    credential_hash: str
    expires_at: int | None = None

    def has_scope(self, scope: Scope) -> bool:
        """Return True when this Principal holds ``scope``."""
        return scope.value in self.scopes

    @property
    def credential_hash_prefix(self) -> str:
        """The first 8 characters of the credential hash.

        Used as a bounded label in the rate-limit and auth log lines.
        Never the credential itself.
        """
        return self.credential_hash[:8]
