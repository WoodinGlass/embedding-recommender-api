"""Filter field allowlist.

The set of filter fields accepted by the recommend endpoint is defined in
``docs/contracts.md`` § 2.1. This module is the single place in the code
where that set lives; the API schemas and the retrieval backends both
reference it. Adding a field is one change here plus one change to
``contracts.md``, in the same PR.

Rationale for the allowlist (as opposed to accepting arbitrary field names)
is in ``docs/adr/0008-filter-strategy.md`` § Filter allowlist: unvetted
field names cannot reach a SQL query, and a typo in a request becomes a
``422`` at the API boundary rather than a silently empty result.
"""

from __future__ import annotations

from collections.abc import Mapping

#: The filter fields the API accepts and the retrieval layer understands.
#: Ordering is stable so that a future canonicalization (for cache keys, for
#: logs) does not depend on dict ordering.
FILTER_FIELDS: tuple[str, ...] = ("brand", "category", "language")

_FILTER_FIELDS_SET: frozenset[str] = frozenset(FILTER_FIELDS)


class UnknownFilterFieldError(ValueError):
    """Raised when a filter contains a key outside :data:`FILTER_FIELDS`."""


def validate_filters(filters: Mapping[str, str] | None) -> dict[str, str] | None:
    """Return a normalized copy of ``filters``, or ``None`` if empty.

    - Rejects any key not in :data:`FILTER_FIELDS` with
      :class:`UnknownFilterFieldError`.
    - Rejects non-string values (the API schemas already produce strings,
      but the retrieval layer is also called from tests and scripts).
    - Drops keys whose value is the empty string, so ``{"category": ""}``
      and ``{"category": None}`` and ``{}`` behave the same way from the
      caller's perspective: no filter.
    - Returns a new dict with keys sorted, so that two logically equal
      filter sets produce the same object for hashing and logging.

    This function is pure: no I/O, no logging, no configuration.
    """
    if not filters:
        return None

    unknown = set(filters.keys()) - _FILTER_FIELDS_SET
    if unknown:
        raise UnknownFilterFieldError(
            f"unknown filter field(s): {sorted(unknown)}; allowed: {list(FILTER_FIELDS)}"
        )

    normalized: dict[str, str] = {}
    for key in FILTER_FIELDS:
        if key not in filters:
            continue
        value = filters[key]
        if not isinstance(value, str):
            raise TypeError(f"filter {key!r} must be a string, got {type(value).__name__}")
        stripped = value.strip()
        if stripped:
            normalized[key] = stripped

    return normalized or None
