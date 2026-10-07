"""Golden set loader and validator.

The golden set is the ground truth that evaluation metrics compare
against. Its format and versioning are fixed by
``docs/adr/0009-golden-set-and-metrics.md`` § 1: one JSON object per line,
one query per line, in a file named ``v1.jsonl`` (or ``v2.jsonl``, ...).
The version is the filename; there is no version field inside the file.

The module is pure: it reads a file and validates it against a caller-
supplied set of known item ids. It does not import a database, does not
read configuration, and does not log.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

#: The default golden set version this project ships.
DEFAULT_GOLDEN_SET_VERSION = "v1"

#: Pattern for a golden set filename. Captures the version.
_FILENAME_RE = re.compile(r"^(?P<version>v\d+)\.jsonl$")

_REQUIRED_FIELDS = frozenset({"query_id", "topic", "seed_item_ids", "relevant_item_ids"})


class GoldenSetError(Exception):
    """Raised when the golden set is malformed or inconsistent."""


@dataclass(frozen=True)
class GoldenQuery:
    """One query in the golden set.

    ``seed_item_ids`` are the items whose embeddings are averaged to form
    the query vector (ADR-0009 § 3). ``relevant_item_ids`` are the items
    that a perfect retrieval should return.
    """

    query_id: str
    topic: str
    seed_item_ids: tuple[str, ...]
    relevant_item_ids: tuple[str, ...]


@dataclass(frozen=True)
class GoldenSet:
    """A parsed, validated golden set."""

    version: str
    path: pathlib.Path
    queries: tuple[GoldenQuery, ...]

    def __len__(self) -> int:
        return len(self.queries)


def version_from_filename(path: pathlib.Path) -> str:
    """Extract the version string from a golden set filename.

    Raises :class:`GoldenSetError` if the filename does not match the
    ``v<N>.jsonl`` pattern. A caller that wants to load a set whose name
    does not follow the pattern has to say so explicitly; a silent
    default would defeat the versioning.
    """
    m = _FILENAME_RE.match(path.name)
    if m is None:
        raise GoldenSetError(f"golden set filename must match 'v<N>.jsonl', got {path.name!r}")
    return m.group("version")


def load_golden_set(
    path: pathlib.Path,
    *,
    known_item_ids: Iterable[str] | None = None,
) -> GoldenSet:
    """Read and validate a golden set from ``path``.

    ``known_item_ids``, when provided, is the set of item ids that exist in
    the catalog. Any seed or relevant id outside this set is an error: a
    query referring to a non-existent item cannot be evaluated, and
    silently ignoring the reference would hide a data problem.

    The validation rules, all from ADR-0009 § 1:

    - Every line is a JSON object.
    - Every object has exactly the required fields (extra fields are
      rejected so a typo does not go unnoticed).
    - ``query_id`` is a non-empty string, unique within the file.
    - ``topic`` is a non-empty string. Not required to be unique (several
      queries can share a topic; the field is for diagnostics).
    - ``seed_item_ids`` and ``relevant_item_ids`` are lists of non-empty
      strings.
    - At least one seed per query (a query with no seeds has no vector).
    - Seeds and relevant ids are disjoint (an item cannot be both the
      source of the query and the answer).
    - All ids are unique within each list.
    - Every id exists in ``known_item_ids`` when that argument is given.
    """
    if not path.is_file():
        raise GoldenSetError(f"golden set not found: {path}")
    version = version_from_filename(path)
    known = frozenset(known_item_ids) if known_item_ids is not None else None

    queries: list[GoldenQuery] = []
    seen_query_ids: set[str] = set()

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetError(f"line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise GoldenSetError(f"line {lineno}: not a JSON object")

        missing = _REQUIRED_FIELDS - set(obj.keys())
        if missing:
            raise GoldenSetError(f"line {lineno}: missing fields {sorted(missing)}")
        extra = set(obj.keys()) - _REQUIRED_FIELDS
        if extra:
            raise GoldenSetError(f"line {lineno}: unexpected fields {sorted(extra)}")

        query_id = obj["query_id"]
        if not isinstance(query_id, str) or not query_id:
            raise GoldenSetError(f"line {lineno}: query_id must be a non-empty string")
        if query_id in seen_query_ids:
            raise GoldenSetError(f"line {lineno}: duplicate query_id {query_id!r}")
        seen_query_ids.add(query_id)

        topic = obj["topic"]
        if not isinstance(topic, str) or not topic:
            raise GoldenSetError(f"line {lineno}: topic must be a non-empty string")

        seeds = _validate_id_list(obj["seed_item_ids"], lineno, "seed_item_ids")
        relevant = _validate_id_list(obj["relevant_item_ids"], lineno, "relevant_item_ids")

        if not seeds:
            raise GoldenSetError(f"line {lineno}: seed_item_ids must contain at least one id")

        overlap = set(seeds) & set(relevant)
        if overlap:
            raise GoldenSetError(f"line {lineno}: seed and relevant ids overlap: {sorted(overlap)}")

        if known is not None:
            unknown = (set(seeds) | set(relevant)) - known
            if unknown:
                raise GoldenSetError(
                    f"line {lineno}: refers to unknown item id(s): {sorted(unknown)}"
                )

        queries.append(
            GoldenQuery(
                query_id=query_id,
                topic=topic,
                seed_item_ids=tuple(seeds),
                relevant_item_ids=tuple(relevant),
            )
        )

    if not queries:
        raise GoldenSetError(f"golden set {path} contains no queries")

    return GoldenSet(version=version, path=path, queries=tuple(queries))


def _validate_id_list(value: object, lineno: int, field: str) -> list[str]:
    """Return a validated list of strings. Raises on malformed input."""
    if not isinstance(value, list):
        raise GoldenSetError(f"line {lineno}: {field} must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise GoldenSetError(f"line {lineno}: {field}[{i}] must be a non-empty string")
        if item in seen:
            raise GoldenSetError(f"line {lineno}: {field} contains duplicate {item!r}")
        seen.add(item)
        out.append(item)
    return out


def default_golden_set_path(
    repo_root: pathlib.Path,
    *,
    version: str = DEFAULT_GOLDEN_SET_VERSION,
) -> pathlib.Path:
    """Return the canonical path for a golden set version."""
    return repo_root / "evaluation" / "golden_set" / f"{version}.jsonl"
