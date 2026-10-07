"""Index identity — the deterministic hash that names an index.

``index_version`` is ``idx-<sha prefix>`` over every input that changes
which items a query returns, in which order. Full rationale in
``docs/adr/0007-index-version-identity.md``; the schema the version is
stored in is in ``docs/adr/0006-pgvector-schema.md``.

The module is pure: it hashes inputs and resolves collisions through an
injected lookup. It does not import a database driver, does not read
config, and does not log. That makes it testable without any of the
infrastructure the caller needs, and keeps the hashing contract visible
in one place.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

#: Default prefix length, in hex characters. See ADR-0007 § Identifier
#: format.
DEFAULT_PREFIX_LENGTH: Final[int] = 8

#: Extended prefix length used on collision. See ADR-0007 § Collision
#: handling.
EXTENDED_PREFIX_LENGTH: Final[int] = 12

#: The version prefix. Kept as a constant so a future change (e.g. to a
#: different family of indexes) is a single edit.
PREFIX: Final[str] = "idx-"


@dataclass(frozen=True)
class IndexIdentityInputs:
    """Every parameter that affects the retrieval result of an index.

    Adding a field here changes the hash of every index built afterwards.
    That is intentional: a new input is a new identity, and the registry
    will refuse to reuse an index built without it. Removing a field is
    also an identity change.

    Excluded by design (see ADR-0007):
    - ``hnsw_ef_search``: a query-time knob, not a build parameter.
    - ``golden_set_version``: evaluation metadata, not a build input.
    """

    model_version: str
    catalog_snapshot: str
    preprocessing_version: str
    metric: str
    hnsw_m: int
    hnsw_ef_construction: int
    pgvector_version: str


def canonical_json(inputs: IndexIdentityInputs) -> str:
    """Return the canonical JSON string that is hashed.

    Sorted keys, no insignificant whitespace, no trailing newline. The
    encoding is stable across Python versions and across machines.
    """
    payload = {
        "model_version": inputs.model_version,
        "catalog_snapshot": inputs.catalog_snapshot,
        "preprocessing_version": inputs.preprocessing_version,
        "metric": inputs.metric,
        "hnsw_m": inputs.hnsw_m,
        "hnsw_ef_construction": inputs.hnsw_ef_construction,
        "pgvector_version": inputs.pgvector_version,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def index_version(
    inputs: IndexIdentityInputs, *, prefix_length: int = DEFAULT_PREFIX_LENGTH
) -> str:
    """Return ``idx-<prefix>`` for ``inputs``.

    ``prefix_length`` is the number of hex characters after ``idx-``. The
    default is :data:`DEFAULT_PREFIX_LENGTH`; the extended length used on
    collision is :data:`EXTENDED_PREFIX_LENGTH`. Any length >= 8 is
    accepted so a test or a future migration can compute a longer prefix
    without duplicating the hashing logic.
    """
    if prefix_length < DEFAULT_PREFIX_LENGTH:
        raise ValueError(f"prefix_length must be >= {DEFAULT_PREFIX_LENGTH}, got {prefix_length}")
    canonical = canonical_json(inputs)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{PREFIX}{digest[:prefix_length]}"


ResolutionStatus = Literal["new", "duplicate"]


def resolve_index_version(
    inputs: IndexIdentityInputs,
    *,
    lookup: Callable[[str], IndexIdentityInputs | None],
) -> tuple[str, ResolutionStatus]:
    """Resolve an ``index_version`` for ``inputs``, extending on collision.

    ``lookup`` is a function that, given a candidate ``index_version``,
    returns the ``IndexIdentityInputs`` recorded for it, or ``None`` if
    the version does not exist. The caller supplies this — usually a
    query against ``index_registry``.

    Returns ``(index_version, status)``:

    - ``("new", ...)`` when the version does not exist. The caller should
      create a registry row with ``status = 'building'``.
    - ``("duplicate", ...)`` when the version exists with **identical**
      inputs. The caller should skip the build (idempotency, ADR-0007).

    A collision at 8 hex with different inputs extends to 12 hex and
    retries. A collision at 12 hex with different inputs raises
    :class:`RuntimeError`; at 48 bits the probability across a portfolio
    project is negligible, and a silent overwrite of a registry row would
    be a bug that is nearly impossible to diagnose.
    """
    for length in (DEFAULT_PREFIX_LENGTH, EXTENDED_PREFIX_LENGTH):
        candidate = index_version(inputs, prefix_length=length)
        existing = lookup(candidate)
        if existing is None:
            return candidate, "new"
        if existing == inputs:
            return candidate, "duplicate"
    raise RuntimeError(
        f"index_version collision beyond {EXTENDED_PREFIX_LENGTH} hex for "
        f"inputs {inputs!r}; manually inspect the registry"
    )
