"""Hashing helpers.

Two families, chosen for different threat models:

- :func:`blake2b_16` returns a 128-bit BLAKE2b hash (32 hex characters).
  Used where the input has enough entropy that dictionary attacks are
  not a concern and the goal is a short, stable identifier: cache keys,
  rate limit keys, the credential hash a ``Principal`` carries. SHA256
  would work; BLAKE2b-128 is half the length at the same collision
  resistance, which halves the memory spent on key names.

- :func:`hmac_sha256` returns a keyed hash. Used for user id hashing
  (ADR-0018) where the input space is small or predictable (an email
  address, a phone number, a sequential id) and an unkeyed hash would
  be reversible against a dictionary. The key is a per-environment
  secret; the version of the key is stored alongside the hash so a
  rotation is a filter, not a data loss.

Both are pure and deterministic. Neither logs; neither reads config.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac

#: BLAKE2b digest size in bytes. 16 bytes = 128 bits = 32 hex chars.
_BLAKE2B_DIGEST_SIZE = 16


def blake2b_16(value: str) -> str:
    """Return the hex-encoded 128-bit BLAKE2b hash of ``value``.

    The output is 32 lowercase hex characters. The input is UTF-8
    encoded; a caller that wants a stable identifier for a request or a
    credential passes the canonical form (sorted keys, no whitespace).
    """
    return hashlib.blake2b(value.encode("utf-8"), digest_size=_BLAKE2B_DIGEST_SIZE).hexdigest()


def hmac_sha256(key: str | bytes, message: str | bytes) -> str:
    """Return the hex-encoded HMAC-SHA256 of ``message`` under ``key``.

    The output is 64 lowercase hex characters. Strings are UTF-8
    encoded; bytes are used as-is. The key must not be empty (an empty
    key weakens the HMAC to an unkeyed hash); callers that read the key
    from config validate it at startup.
    """
    if not key:
        raise ValueError("hmac_sha256 requires a non-empty key")
    key_b = key.encode("utf-8") if isinstance(key, str) else key
    msg_b = message.encode("utf-8") if isinstance(message, str) else message
    return _hmac.new(key_b, msg_b, hashlib.sha256).hexdigest()
