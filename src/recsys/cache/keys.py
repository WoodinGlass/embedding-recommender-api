"""Cache key derivation (ADR-0015).

The key schema is::

    cache:v1:{endpoint}:{blake2b-16(canonical)}[:{variant}]

Each segment has a reason:

- ``cache`` — a namespace prefix. It lets an operator ``SCAN`` or
  ``DEL`` every cached entry without touching the rate-limit keys
  that share the same Redis instance.
- ``v1`` — a key schema version. A change to what goes into the hash,
  or how the hash is computed, bumps this. The old namespace drains
  by TTL and is removed; there is no in-place migration.
- ``{endpoint}`` — ``recommend`` or ``similar``. Different endpoints
  have different TTLs and different cache eligibility. The set is
  closed so a typo cannot create an unreachable entry.
- ``{blake2b-16}`` — 32 hex characters, the BLAKE2b-128 digest of the
  canonical request. BLAKE2b over SHA256 because the key is stored
  per entry and 16 bytes halves the name memory at the same
  collision resistance.
- ``[:{variant}]`` — the experiment variant, appended as a suffix,
  not folded into the hash. Two variants of one query are two keys,
  and a ``SCAN`` for ``cache:v1:recommend:*:control`` finds every
  control entry. Folding the variant into the hash makes that
  impossible.

The hash input is the canonical request — index version, endpoint,
k, sorted filters, sorted seed item ids. The ``user_id`` is **not**
part of it: personalization at this project's scale comes from the
seed items, and adding the user id would multiply the key space by
the user count and push the hit rate toward zero for no behavioral
change.

All functions here are pure: no IO, no config, no logging. The
caller is responsible for passing values that have already been
validated (filters from ``FILTER_FIELDS``, k within bounds).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final

from recsys.security.hashing import blake2b_16

# ADR-0015 amendment: the prefix is namespaced so it cannot
# collide with the rate limiter (`recsys:rl:*`) on a shared Redis.
_NAMESPACE: Final[str] = "recsys:cache"
_SCHEMA_VERSION: Final[str] = "v1"

#: The set of endpoints the key builder accepts. A closed set is a
#: cheap guard against a typo becoming an entry nothing will ever
#: read (a wrong endpoint produces a valid-looking key that no other
#: caller will compute).
ALLOWED_ENDPOINTS: Final[frozenset[str]] = frozenset({"recommend", "similar"})


def _canonical_payload(
    *,
    index_version: str,
    endpoint: str,
    k_max: int,
    filters: Mapping[str, str] | None,
    seed_item_ids: Sequence[str],
) -> str:
    """Return a stable JSON encoding of the request.

    ``k_max`` rather than ``k``: the cache stores a fixed-size
    candidate window and the handler slices to the request's ``k``
    (ADR-0015 amendment). A request for k=5 and one for k=20 share
    one entry, because the answer to the smaller is a prefix of the
    larger.

    ``seed_item_ids`` are **sorted and deduplicated**. Sorting makes
    ``[a, b]`` and ``[b, a]`` the same key; deduplication makes
    ``[a, a, b]`` and ``[a, b]`` the same key. Without either, a
    client that reorders or repeats seeds produces a permanent miss.
    """
    canonical_filters = dict(sorted((filters or {}).items()))
    canonical_seeds = sorted(set(seed_item_ids))
    return json.dumps(
        {
            "index_version": index_version,
            "endpoint": endpoint,
            "k_max": k_max,
            "filters": canonical_filters,
            "seed_item_ids": canonical_seeds,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def build_cache_key(
    *,
    index_version: str,
    endpoint: str,
    k_max: int,
    filters: Mapping[str, str] | None = None,
    seed_item_ids: Sequence[str] = (),
    variant: str | None = None,
) -> str:
    """Return the Redis key for this request.

    ``k_max`` (not ``k``) is the candidate window the cache holds;
    see ``_canonical_payload``. The variant suffix is appended only
    when ``variant`` is not ``None``; a caller with no experiment
    running produces the shorter key.
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"endpoint must be one of {sorted(ALLOWED_ENDPOINTS)}, got {endpoint!r}")
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")
    if not index_version:
        raise ValueError("index_version must be a non-empty string")

    payload = _canonical_payload(
        index_version=index_version,
        endpoint=endpoint,
        k_max=k_max,
        filters=filters,
        seed_item_ids=seed_item_ids,
    )
    digest = blake2b_16(payload)
    base = f"{_NAMESPACE}:{_SCHEMA_VERSION}:{endpoint}:{digest}"
    if variant is None:
        return base
    return f"{base}:{variant}"
