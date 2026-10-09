"""Fallback chain (ADR-0020).

Public surface:

- :class:`FallbackResult` — the source label and the items a fallback
  tier produced.
- :func:`serve_fallback` — run tier 3 (DB snapshot) then tier 4
  (in-memory cache); return the first non-empty result, or ``None``
  when both are empty (the caller then serves a 503 with
  ``meta.source="none"``).
"""

from recsys.fallback.chain import FallbackResult, serve_fallback

__all__ = [
    "FallbackResult",
    "serve_fallback",
]
