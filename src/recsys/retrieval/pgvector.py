"""pgvector HNSW backend.

Production retrieval path. Reads embeddings from the ``embedding`` table
and returns the nearest items by cosine distance, optionally filtered by
metadata joined from the ``item`` table.

Version detection and the two query strategies (iterative scan on
pgvector >= 0.8, over-fetch fallback on older versions) follow
``docs/adr/0008-filter-strategy.md``. The backend is synchronous and takes
``NDArray[np.float32]`` per ``docs/adr/0012-backend-abstraction.md``.

The backend does not manage the connection lifecycle. The caller — a
build script, the evaluation runner, or (from M3) the serving layer —
creates the connection and passes it in. This keeps the backend a pure
strategy over a database, and makes it constructible from a fake
connection in unit tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from recsys.monitoring.logging import get_logger
from recsys.retrieval.filters import FILTER_FIELDS, validate_filters

log = get_logger(__name__)

#: Bounded scan for iterative_scan. See ADR-0008 § Filter selectivity.
DEFAULT_MAX_SCAN_TUPLES: Final[int] = 20_000

#: Over-fetch multiplier for pgvector < 0.8. See ADR-0008.
FALLBACK_OVERFETCH: Final[int] = 4

#: Version at which hnsw.iterative_scan became available.
ITERATIVE_SCAN_MIN_VERSION: Final[tuple[int, int]] = (0, 8)

#: Per-process flag: log the detected version exactly once.
_VERSION_LOGGED: bool = False

#: Cosine distance between L2-normalized vectors is in [0, 2]; the schema
#: requires scores in [0, 1]. Clamping does not affect the ordering.
_SCORE_MIN: Final[float] = 0.0
_SCORE_MAX: Final[float] = 1.0


@dataclass(frozen=True)
class PgvectorVersion:
    """Parsed pgvector extension version."""

    raw: str
    major: int
    minor: int

    @property
    def supports_iterative_scan(self) -> bool:
        return (self.major, self.minor) >= ITERATIVE_SCAN_MIN_VERSION


def _parse_extversion(raw: str) -> PgvectorVersion:
    """Parse ``extversion`` from ``pg_extension``.

    Handles ``"0.8.0"``, ``"0.7.4"``, ``"0.8"``, and pre-release suffixes
    like ``"0.8.0-beta"`` by taking the leading ``major.minor``. A string
    with fewer than two numeric components is treated as an unparseable
    version and defaults to ``(0, 0)``, which does not support iterative
    scan — the conservative choice.
    """
    head = raw.strip().split("-", 1)[0]
    parts = head.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return PgvectorVersion(raw=raw, major=0, minor=0)
    return PgvectorVersion(raw=raw, major=major, minor=minor)


def _coerce_vector(raw: Any) -> NDArray[np.float32]:
    """Coerce a raw ``embedding.vector`` column value to a float32 array.

    The driver's representation depends on how the connection is
    configured. pgvector ships an adapter (``pgvector.psycopg``) that
    decodes the column to ``numpy.ndarray`` when registered; without it,
    psycopg hands back the column's text form, ``"[0.1, 0.2, ...]"``. The
    evaluation script does not register the adapter because the
    connection is shared with other callers and the registration is a
    global side effect. This helper handles all three representations
    that occur in practice:

    - ``numpy.ndarray`` — the adapter is registered.
    - ``bytes`` / ``bytearray`` — a driver that returns the wire form
      undecoded.
    - ``str`` — the default; a bracketed comma-separated list.
    - a Python list — a JSON-decoded value from a driver that never
      sees the pgvector adapter.

    A malformed value raises ``ValueError`` with the underlying reason,
    which the caller turns into a per-arm error rather than a crash.
    """
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float32, copy=False)
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("ascii")
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return np.zeros(0, dtype=np.float32)
        # ``np.fromstring`` is deprecated; a comprehension is explicit
        # and fast enough for the 384-float payload this handles.
        return np.array([float(x) for x in s.split(",")], dtype=np.float32)
    return np.asarray(raw, dtype=np.float32)


def _vector_to_literal(vector: NDArray[np.float32]) -> str:
    """Render a float32 vector as the literal pgvector accepts.

    ``[0.1,0.2,...]`` — the format pgvector parses. Values are formatted
    with ``repr`` on their Python ``float`` conversion, which preserves the
    float32 value exactly (Python's float has enough precision for any
    float32).
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


@dataclass(frozen=True)
class _SearchPlan:
    """Everything needed to execute a search against a specific index."""

    sql: str
    filter_field_order: tuple[str, ...]


def _build_search_plan(
    *,
    filters: Mapping[str, str] | None,
    with_vectors: bool = False,
) -> _SearchPlan:
    """Build the SQL and the order of filter values.

    The SQL uses named parameters (``%(name)s``), never interpolation. The
    filter field order returned here matches the order the caller must
    supply values in ``%(f_<field>)s``.

    ``with_vectors=True`` adds the raw embedding column to the ``SELECT``
    list. The cost is one extra column per returned row (384 float32, so
    ~1.5 KB per candidate); the benefit is that MMR does not need a
    second round trip. The caller asks for it only when the re-ranker's
    MMR step is enabled (ADR-0016 § Step 3).
    """
    filter_clauses: list[str] = []
    filter_fields: list[str] = []
    if filters:
        for field in FILTER_FIELDS:
            if field in filters:
                filter_clauses.append(f"AND i.{field} = %(f_{field})s")
                filter_fields.append(field)

    where_filters = ""
    if filter_clauses:
        where_filters = "\n  ".join(filter_clauses)

    vector_column = ",\n       e.vector AS vector" if with_vectors else ""
    sql = f"""
SELECT e.item_id,
       (e.vector <=> %(query_vec)s::vector) AS distance{vector_column}
FROM embedding e
JOIN item i ON i.item_id = e.item_id
WHERE e.index_version = %(index_version)s
  {where_filters}
ORDER BY e.vector <=> %(query_vec)s::vector
LIMIT %(limit)s
""".strip()  # noqa: S608
    return _SearchPlan(sql=sql, filter_field_order=tuple(filter_fields))


class PgvectorBackend:
    """pgvector HNSW backend.

    Satisfies :class:`recsys.retrieval.base.IndexBackend`. The caller owns
    the psycopg connection and passes it in. The backend detects the
    pgvector extension version once, at construction, and uses the two
    query strategies from ADR-0008 accordingly.
    """

    name: str = "pgvector"

    def __init__(
        self,
        connection: Any,
        *,
        active_index_version: str,
        hnsw_ef_search: int,
        max_scan_tuples: int = DEFAULT_MAX_SCAN_TUPLES,
        pgvector_version: Any | None = None,
    ) -> None:
        """Construct a backend bound to a live connection.

        ``connection`` is a psycopg connection (sync). ``active_index_version``
        is the ``index_registry.index_version`` to query against; the caller
        is expected to have read it from the registry (see
        :meth:`from_registry`). ``hnsw_ef_search`` is the query-time knob
        (``SET LOCAL hnsw.ef_search``), read from config per
        ``docs/contracts.md`` § 3.
        """
        # Duck-typed interface check: the backend needs a connection with
        # ``cursor()`` and ``transaction()``, not specifically psycopg.
        # This makes the backend constructible from a fake connection in
        # unit tests (which run in Colab without psycopg) and from a real
        # psycopg connection in production. The psycopg dependency lives in
        # the scripts that open connections (build_index, promote_index,
        # the M3 API startup), not here. See ADR-0012.
        if not (hasattr(connection, "cursor") and hasattr(connection, "transaction")):
            raise TypeError(
                "connection must provide cursor() and transaction(); "
                f"got {type(connection).__name__}"
            )
        if not active_index_version:
            raise ValueError("active_index_version must be non-empty")
        if hnsw_ef_search < 1:
            raise ValueError(f"hnsw_ef_search must be >= 1, got {hnsw_ef_search}")
        if max_scan_tuples < 1:
            raise ValueError(f"max_scan_tuples must be >= 1, got {max_scan_tuples}")

        self._connection = connection
        self._active_index_version = active_index_version
        self._hnsw_ef_search = hnsw_ef_search
        self._max_scan_tuples = max_scan_tuples

        # ``pgvector_version`` is prefetched at startup (ADR-0015
        # amendment). When None (unit tests, or the ``scripts/``
        # paths that do not go through the app), detect it — one
        # query, acceptable off the hot path.
        if pgvector_version is None:
            self._pgvector_version: PgvectorVersion = self._detect_pgvector_version()
        else:
            self._pgvector_version = pgvector_version
        self._log_version_once()

    # ------------------------------------------------------------------ props
    @property
    def active_index_version(self) -> str:
        return self._active_index_version

    @property
    def pgvector_version(self) -> PgvectorVersion:
        return self._pgvector_version

    # ---------------------------------------------------------------- version
    def _detect_pgvector_version(self) -> PgvectorVersion:
        with self._connection.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "pgvector extension is not installed in this database; "
                "run `CREATE EXTENSION vector;` (see migrations/0001)."
            )
        raw = str(row[0])
        parsed = _parse_extversion(raw)
        return parsed

    def _log_version_once(self) -> None:
        global _VERSION_LOGGED
        if _VERSION_LOGGED:
            return
        _VERSION_LOGGED = True
        log.info(
            "retrieval.pgvector.version",
            detected=self._pgvector_version.raw,
            supports_iterative_scan=self._pgvector_version.supports_iterative_scan,
        )
        if not self._pgvector_version.supports_iterative_scan:
            log.warning(
                "retrieval.pgvector.old_version",
                detected=self._pgvector_version.raw,
                required_for_iterative_scan="0.8",
                fallback="post_filter_k_multiplier",
                multiplier=FALLBACK_OVERFETCH,
            )

    # ------------------------------------------------------------------ ready
    def is_ready(self) -> bool:
        """Return ``True`` when the connection is usable and an active row
        exists in ``index_registry`` for this backend's version.
        """
        try:
            if self._connection.closed:
                return False
            with self._connection.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM index_registry WHERE index_version = %s",
                    (self._active_index_version,),
                )
                return cur.fetchone() is not None
        except Exception:
            # Any failure here means "not ready"; the caller (readiness
            # endpoint in M3) treats a False as drain-this-instance.
            return False

    # ----------------------------------------------------------------- search
    def search(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Run an HNSW search, applying filters when present.

        On pgvector >= 0.8, filtered queries use
        ``hnsw.iterative_scan = 'strict_order'`` bounded by
        ``hnsw.max_scan_tuples``. On older versions, the query over-fetches
        by :data:`FALLBACK_OVERFETCH` and returns the first ``k`` — the
        WHERE clause still applies, but the effective recall after a
        selective filter is bounded by the number of HNSW candidates the
        pre-0.8 planner is willing to produce.
        """
        rows = self._execute(vector=vector, k=k, filters=filters, with_vectors=False)
        return self._rows_to_pairs(rows, k=k)

    def search_with_vectors(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float, NDArray[np.float32]]]:
        """Same as :meth:`search`, with the candidate vector on each result.

        Used by the re-ranker's MMR step (ADR-0016 § Step 3): MMR computes
        pairwise cosine similarity on the candidate vectors, and having
        them on the result avoids a second round trip. The cost is one
        extra column per row (~1.5 KB per candidate at 384-dim float32),
        which is why this is a separate method rather than the default.
        """
        rows = self._execute(vector=vector, k=k, filters=filters, with_vectors=True)
        result: list[tuple[str, float, NDArray[np.float32]]] = []
        for item_id, distance, raw_vec in rows[:k]:
            similarity = self._clamp_similarity(1.0 - float(distance))
            vec = _coerce_vector(raw_vec)
            result.append((str(item_id), similarity, vec))
        return result

    # ------------------------------------------------------------ internals
    def _execute(
        self,
        *,
        vector: NDArray[np.float32],
        k: int,
        filters: Mapping[str, str] | None,
        with_vectors: bool,
    ) -> list[tuple[Any, ...]]:
        """Run the search query and return raw rows.

        ``search`` and ``search_with_vectors`` share this; the only
        difference is whether the SELECT list includes the vector column.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if vector.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {vector.shape}")
        if vector.dtype != np.float32:
            raise ValueError(f"vector must be float32, got {vector.dtype}")
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError(
                f"query vector must be L2-normalized (|v| = {norm:.6f}); "
                f"normalizing is the caller's responsibility per ADR-0009 § 3"
            )

        normalized_filters = validate_filters(filters)
        plan = _build_search_plan(filters=normalized_filters, with_vectors=with_vectors)

        supports_scan = self._pgvector_version.supports_iterative_scan
        fetch_k = k if supports_scan else k * FALLBACK_OVERFETCH

        params: dict[str, Any] = {
            "query_vec": _vector_to_literal(vector),
            "index_version": self._active_index_version,
            "limit": fetch_k,
        }
        if normalized_filters is not None:
            for field in plan.filter_field_order:
                params[f"f_{field}"] = normalized_filters[field]

        with self._connection.transaction():
            if normalized_filters is not None and supports_scan:
                with self._connection.cursor() as cur:
                    # `SET` does not accept bound parameters; the function
                    # form `set_config(name, value, is_local)` does. The
                    # third argument `true` scopes the setting to the
                    # current transaction, matching `SET LOCAL` semantics.
                    cur.execute(
                        "SELECT set_config('hnsw.iterative_scan', %s, true)",
                        ("strict_order",),
                    )
                    cur.execute(
                        "SELECT set_config('hnsw.max_scan_tuples', %s, true)",
                        (str(self._max_scan_tuples),),
                    )
            with self._connection.cursor() as cur:
                cur.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(self._hnsw_ef_search),),
                )
                cur.execute(plan.sql, params)
                rows = list(cur.fetchall())

        if normalized_filters is not None and not supports_scan and len(rows) < k:
            log.warning(
                "retrieval.pgvector.fallback",
                reason="pgvector_lt_0_8",
                detected=self._pgvector_version.raw,
                requested_k=k,
                effective_k=fetch_k,
                returned=len(rows),
            )
        return rows

    @staticmethod
    def _clamp_similarity(similarity: float) -> float:
        if similarity < _SCORE_MIN:
            return _SCORE_MIN
        if similarity > _SCORE_MAX:
            return _SCORE_MAX
        return similarity

    def _rows_to_pairs(
        self,
        rows: list[tuple[Any, ...]],
        *,
        k: int,
    ) -> list[tuple[str, float]]:
        result: list[tuple[str, float]] = []
        for item_id, distance in rows[:k]:
            result.append((str(item_id), self._clamp_similarity(1.0 - float(distance))))
        return result

    # -------------------------------------------------------------- factories
    @classmethod
    def from_registry(
        cls,
        connection: Any,
        *,
        hnsw_ef_search: int,
        max_scan_tuples: int = DEFAULT_MAX_SCAN_TUPLES,
    ) -> PgvectorBackend:
        """Construct from the active index in ``index_registry``.

        Raises :class:`RuntimeError` if there is no row with
        ``status = 'active'``. The partial unique index on
        ``status = 'active'`` guarantees at most one such row.
        """
        with connection.cursor() as cur:
            cur.execute("SELECT index_version FROM index_registry WHERE status = 'active'")
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "no active index in index_registry; run "
                "`python scripts/build_index.py --promote` first"
            )
        return cls(
            connection,
            active_index_version=str(row[0]),
            hnsw_ef_search=hnsw_ef_search,
            max_scan_tuples=max_scan_tuples,
        )
