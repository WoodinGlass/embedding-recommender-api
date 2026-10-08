"""Cryptographic helpers used across the API.

Two functions, one module: :func:`blake2b_16` for cache and rate-limit
keys (short, non-reversible hashes over high-entropy input) and
:func:`hmac_sha256` for user id hashing (ADR-0018), where the input
space is small and a plain hash would be dictionary-reversible.
"""

from recsys.security.hashing import blake2b_16, hmac_sha256

__all__ = ["blake2b_16", "hmac_sha256"]
