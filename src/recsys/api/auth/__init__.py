"""Authentication and authorization (ADR-0013).

Public surface:

- :class:`Principal`, :class:`PrincipalKind`, :class:`Scope` — the value
  that flows through a request after a credential is validated.
- :func:`validate_api_key` — an API key's hash comparison.
- :func:`validate_jwt` — a JWT's signature and claim validation.
- :func:`hash_api_key` — the operator tooling's hashing function.
"""

from recsys.api.auth.api_key import hash_api_key, validate_api_key
from recsys.api.auth.jwt import JwtError, validate_jwt
from recsys.api.auth.principal import Principal, PrincipalKind, Scope

__all__ = [
    "JwtError",
    "Principal",
    "PrincipalKind",
    "Scope",
    "hash_api_key",
    "validate_api_key",
    "validate_jwt",
]
