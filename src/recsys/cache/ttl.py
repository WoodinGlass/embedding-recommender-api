"""Cache TTLs and the negative-cache sentinel (ADR-0015).

Two decisions live here:

- **Jitter.** The actual TTL is the configured value times a uniform
  random factor in ``[0.9, 1.1)``. Without jitter, a batch of
  entries written in the same second expires in the same second, and
  the first request after expiry pays for every entry at once (a
  stampede). Jitter spreads the expiry across a window twice as wide
  as the TTL's precision, which is cheap and effective.

- **Negative caching.** A miss that resolves to "no such item" is
  stored as a sentinel for the negative TTL. A caller who probes a
  nonexistent id repeatedly does not hit the retrieval path on every
  request. The sentinel is bytes, not a string: the Redis client is
  configured with ``decode_responses=False`` (a Python bytes
  round-trip), and a str sentinel would silently never match. The
  value is chosen so it cannot collide with a legitimate cached
  payload: a real value is a serialized list of ``(item_id, score)``
  pairs and always starts with ``[``.
"""

from __future__ import annotations

import secrets
from random import Random
from typing import Final

#: Sentinel stored under a negative-cache key. Bytes because the
#: Redis client uses ``decode_responses=False``; see the module
#: docstring.
NEGATIVE_SENTINEL: Final[bytes] = b"\x00negative"

#: Lower and upper bounds of the jitter factor. The upper bound is
#: exclusive (``Random.uniform`` is documented as inclusive, but the
#: exact bound is unobservable at integer TTLs).
_JITTER_LOW: Final[float] = 0.9
_JITTER_HIGH: Final[float] = 1.1


def _default_rng() -> Random:
    """Return a fresh RNG.

    ``secrets.randbits`` seeds a ``Random`` with OS entropy. Creating
    a new instance per call keeps the function pure with respect to
    process state — a shared instance would tie the TTL sequence to
    whatever else was drawing from it.
    """
    # S311 targets randomness used for security (tokens, nonces).
    # Jittering a cache TTL is not a security use: the TTL is public,
    # an adversary gains nothing from predicting it, and a
    # cryptographic source would cost entropy for no benefit.
    return Random(secrets.randbits(64))  # noqa: S311


def jittered_ttl(base_seconds: int, *, rng: Random | None = None) -> int:
    """Return ``base_seconds`` multiplied by a uniform factor in [0.9, 1.1).

    The result is an integer number of seconds and never below 1: a
    TTL of 0 is Redis's signal for "no expiry", which would leak an
    entry forever. ``rng`` is injectable so a test can pin the
    factor; production passes ``None`` and gets a fresh RNG.
    """
    if base_seconds < 1:
        raise ValueError(f"base_seconds must be >= 1, got {base_seconds}")
    source = rng if rng is not None else _default_rng()
    factor = source.uniform(_JITTER_LOW, _JITTER_HIGH)
    return max(1, int(base_seconds * factor))
