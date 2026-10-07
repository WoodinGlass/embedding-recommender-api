"""Unit tests for :class:`PgvectorBackend`.

The backend is a strategy over a database connection. A fake connection
that records the SQL it receives and returns canned rows exercises the
whole contract without a live database, which means these tests run on
Colab (no Docker) and in CI alike. The integration tests that open a
real pgvector connection live in ``tests/integration/test_pgvector.py``
and are marked ``integration``.

Not covered here (deliberately): the HNSW planner's behavior, the
``hnsw.iterative_scan`` GUC's effect on recall, and the exact scores
pgvector returns. Those are properties of the database, not of this
module.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.retrieval.pgvector import (
    DEFAULT_MAX_SCAN_TUPLES,
    FALLBACK_OVERFETCH,
    PgvectorBackend,
    PgvectorVersion,
    _build_search_plan,
    _parse_extversion,
    _vector_to_literal,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# fake psycopg connection
# --------------------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self, parent: _FakeConnection) -> None:
        self._parent = parent
        self._fetch_buffer: list[tuple[Any, ...]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._parent.executed.append((sql, params))
        self._fetch_buffer = list(self._parent.respond(sql))

    def fetchone(self) -> tuple[Any, ...] | None:
        if not self._fetch_buffer:
            return None
        return self._fetch_buffer.pop(0)

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows: list[tuple[Any, ...]] = list(self._fetch_buffer)
        self._fetch_buffer.clear()
        return rows


class _FakeConnection:
    """Records executed SQL and returns scripted rows.

    ``responses`` is a callable that takes a SQL string and returns the
    rows the next ``fetchone`` / ``fetchall`` should see. A default
    response (extension version query and index lookup) is provided.
    """

    def __init__(
        self,
        *,
        pgvector_version: str = "0.8.0",
        active_index_version: str = "idx-test",
        search_rows: list[tuple[str, float]] | None = None,
        responses: Any = None,
    ) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._version = pgvector_version
        self._active = active_index_version
        self._search_rows = search_rows if search_rows is not None else []
        self._responses = responses
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield

    def respond(self, sql: str) -> list[tuple[Any, ...]]:
        if self._responses is not None:
            custom: list[tuple[Any, ...]] | None = self._responses(sql)
            if custom is not None:
                return custom
        s = sql.strip().upper()
        if s.startswith("SELECT EXTVERSION"):
            return [(self._version,)]
        if s.startswith("SELECT 1 FROM INDEX_REGISTRY"):
            return [(1,)]
        if s.startswith("SELECT INDEX_VERSION FROM INDEX_REGISTRY"):
            return [(self._active,)]
        # Default: the search query itself.
        return list(self._search_rows)


# --------------------------------------------------------------------------- #
# version parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "major", "minor", "supports"),
    [
        ("0.8.0", 0, 8, True),
        ("0.8", 0, 8, True),
        ("0.8.1", 0, 8, True),
        ("0.8.0-beta", 0, 8, True),
        ("0.9.0", 0, 9, True),
        ("1.0.0", 1, 0, True),
        ("0.7.4", 0, 7, False),
        ("0.7", 0, 7, False),
        ("0.6.2", 0, 6, False),
        ("garbage", 0, 0, False),
        ("", 0, 0, False),
    ],
)
def test_parse_extversion(raw: str, major: int, minor: int, supports: bool) -> None:
    v = _parse_extversion(raw)
    assert v.raw == raw
    assert v.major == major
    assert v.minor == minor
    assert v.supports_iterative_scan is supports


def test_pgvector_version_is_frozen_dataclass() -> None:
    from dataclasses import FrozenInstanceError

    v = PgvectorVersion(raw="0.8.0", major=0, minor=8)
    with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
        v.major = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# vector literal
# --------------------------------------------------------------------------- #
def test_vector_literal_format() -> None:
    v = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    literal = _vector_to_literal(v)
    assert literal.startswith("[")
    assert literal.endswith("]")
    assert literal == "[0.5,-0.25,1.0]"


def test_vector_literal_float32_precision() -> None:
    v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    literal = _vector_to_literal(v)
    # Each value round-trips through Python float, which preserves float32
    # exactly. The literal must not truncate to fewer digits.
    parts = literal[1:-1].split(",")
    assert len(parts) == 3
    for part in parts:
        float(part)  # parses


# --------------------------------------------------------------------------- #
# SQL plan
# --------------------------------------------------------------------------- #
def test_plan_without_filters_has_no_where_clause() -> None:
    plan = _build_search_plan(filters=None)
    assert "AND i." not in plan.sql
    assert plan.filter_field_order == ()


def test_plan_with_filters_uses_named_parameters() -> None:
    plan = _build_search_plan(filters={"category": "books"})
    assert "AND i.category = %(f_category)s" in plan.sql
    # No f-string interpolation: the value never appears in the SQL text.
    assert "books" not in plan.sql


def test_plan_with_multiple_filters_orders_fields() -> None:
    plan = _build_search_plan(filters={"language": "en", "category": "books"})
    # Order follows FILTER_FIELDS = (brand, category, language); the input
    # dict order is ignored.
    assert plan.filter_field_order == ("category", "language")


def test_plan_never_interpolates_filter_values() -> None:
    plan = _build_search_plan(filters={"brand": "'; DROP TABLE item; --"})
    assert "DROP" not in plan.sql
    assert "%(f_brand)s" in plan.sql


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_construction_rejects_empty_index_version() -> None:
    with pytest.raises(ValueError, match="active_index_version"):
        PgvectorBackend(
            _FakeConnection(),
            active_index_version="",
            hnsw_ef_search=100,
        )


def test_construction_rejects_bad_ef_search() -> None:
    with pytest.raises(ValueError, match="hnsw_ef_search"):
        PgvectorBackend(
            _FakeConnection(),
            active_index_version="idx-1",
            hnsw_ef_search=0,
        )


def test_construction_rejects_bad_max_scan_tuples() -> None:
    with pytest.raises(ValueError, match="max_scan_tuples"):
        PgvectorBackend(
            _FakeConnection(),
            active_index_version="idx-1",
            hnsw_ef_search=100,
            max_scan_tuples=0,
        )


def test_construction_queries_extension_version() -> None:
    conn = _FakeConnection(pgvector_version="0.7.4")
    be = PgvectorBackend(conn, active_index_version="idx-1", hnsw_ef_search=100)
    assert be.pgvector_version == PgvectorVersion("0.7.4", 0, 7)
    # The version query was executed exactly once, at construction.
    version_queries = [q for q, _ in conn.executed if "extversion" in q.lower()]
    assert len(version_queries) == 1


def test_construction_fails_when_extension_missing() -> None:
    def no_ext(sql: str) -> list[tuple[Any, ...]] | None:
        if sql.strip().upper().startswith("SELECT EXTVERSION"):
            return []
        return None

    with pytest.raises(RuntimeError, match="pgvector extension is not installed"):
        PgvectorBackend(
            _FakeConnection(responses=no_ext),
            active_index_version="idx-1",
            hnsw_ef_search=100,
        )


# --------------------------------------------------------------------------- #
# is_ready
# --------------------------------------------------------------------------- #
def test_is_ready_when_active_index_exists() -> None:
    be = PgvectorBackend(_FakeConnection(), active_index_version="idx-1", hnsw_ef_search=100)
    assert be.is_ready() is True


def test_is_ready_false_when_connection_closed() -> None:
    conn = _FakeConnection()
    be = PgvectorBackend(conn, active_index_version="idx-1", hnsw_ef_search=100)
    conn.closed = True
    assert be.is_ready() is False


# --------------------------------------------------------------------------- #
# search: validation
# --------------------------------------------------------------------------- #
def _backend(
    *,
    pgvector_version: str = "0.8.0",
    search_rows: list[tuple[str, float]] | None = None,
) -> tuple[PgvectorBackend, _FakeConnection]:
    conn = _FakeConnection(pgvector_version=pgvector_version, search_rows=search_rows)
    be = PgvectorBackend(conn, active_index_version="idx-1", hnsw_ef_search=100)
    conn.executed.clear()  # drop the version-query from the log
    return be, conn


def _unit_query() -> NDArray[np.float32]:
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_search_rejects_non_positive_k() -> None:
    be, _ = _backend()
    with pytest.raises(ValueError, match="k must be positive"):
        be.search(vector=_unit_query(), k=0)


def test_search_rejects_2d_query() -> None:
    be, _ = _backend()
    with pytest.raises(ValueError, match="must be 1-D"):
        be.search(vector=np.zeros((1, 3), dtype=np.float32), k=1)


def test_search_rejects_non_float32() -> None:
    be, _ = _backend()
    with pytest.raises(ValueError, match="float32"):
        be.search(vector=np.array([1.0, 0.0, 0.0], dtype=np.float64), k=1)


def test_search_rejects_non_normalized_query() -> None:
    be, _ = _backend()
    v = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="L2-normalized"):
        be.search(vector=v, k=1)


# --------------------------------------------------------------------------- #
# search: query execution
# --------------------------------------------------------------------------- #
def test_search_executes_ef_search_setting() -> None:
    be, conn = _backend()
    be.search(vector=_unit_query(), k=3)
    sqls = [q for q, _ in conn.executed]
    assert any("set_config" in q and "hnsw.ef_search" in q for q in sqls), sqls


def test_search_does_not_set_iterative_scan_without_filters() -> None:
    be, conn = _backend(pgvector_version="0.8.0")
    be.search(vector=_unit_query(), k=3)
    sqls = " ".join(q for q, _ in conn.executed)
    assert "iterative_scan" not in sqls


def test_search_sets_iterative_scan_with_filters_on_08() -> None:
    be, conn = _backend(pgvector_version="0.8.0")
    be.search(vector=_unit_query(), k=3, filters={"category": "books"})
    # "strict_order" is passed as a bound parameter to set_config, not
    # interpolated into the SQL text. Check the function names in the SQL
    # and the parameter values separately.
    sqls = " ".join(q for q, _ in conn.executed)
    assert "hnsw.iterative_scan" in sqls
    assert "hnsw.max_scan_tuples" in sqls

    values_seen: list[str] = []
    for _q, params in conn.executed:
        if isinstance(params, tuple):
            for v in params:
                values_seen.append(str(v))
    assert "strict_order" in values_seen, values_seen


def test_search_does_not_set_iterative_scan_on_07() -> None:
    be, conn = _backend(pgvector_version="0.7.4")
    be.search(vector=_unit_query(), k=3, filters={"category": "books"})
    sqls = " ".join(q for q, _ in conn.executed)
    assert "iterative_scan" not in sqls


def test_search_uses_k_as_limit_when_iterative_scan_available() -> None:
    be, conn = _backend(pgvector_version="0.8.0")
    be.search(vector=_unit_query(), k=5, filters={"category": "books"})
    search_calls = [
        params
        for q, params in conn.executed
        if params is not None and isinstance(params, dict) and "limit" in params
    ]
    assert search_calls, conn.executed
    assert search_calls[-1]["limit"] == 5


def test_search_uses_k_times_multiplier_on_07() -> None:
    be, conn = _backend(pgvector_version="0.7.4")
    be.search(vector=_unit_query(), k=5, filters={"category": "books"})
    search_calls = [
        params
        for q, params in conn.executed
        if params is not None and isinstance(params, dict) and "limit" in params
    ]
    assert search_calls
    assert search_calls[-1]["limit"] == 5 * FALLBACK_OVERFETCH


def test_search_returns_similarity_as_one_minus_distance() -> None:
    # Distance 0.0 → similarity 1.0; distance 0.3 → similarity 0.7.
    be, _ = _backend(search_rows=[("i_1", 0.0), ("i_2", 0.3), ("i_3", 0.9)])
    results = be.search(vector=_unit_query(), k=3)
    assert results == [("i_1", 1.0), ("i_2", pytest.approx(0.7)), ("i_3", pytest.approx(0.1))]


def test_search_clamps_negative_similarity_to_zero() -> None:
    # Cosine distance can exceed 1.0 for dissimilar vectors; similarity
    # would be negative. The contract requires [0, 1].
    be, _ = _backend(search_rows=[("i_1", 1.4)])
    results = be.search(vector=_unit_query(), k=1)
    assert results == [("i_1", 0.0)]


def test_search_clamps_positive_similarity_to_one() -> None:
    be, _ = _backend(search_rows=[("i_1", -0.1)])
    results = be.search(vector=_unit_query(), k=1)
    assert results == [("i_1", 1.0)]


def test_search_truncates_results_to_k() -> None:
    be, _ = _backend(search_rows=[("i_1", 0.0), ("i_2", 0.1), ("i_3", 0.2)])
    results = be.search(vector=_unit_query(), k=2)
    assert len(results) == 2


def test_search_passes_index_version_and_filters_as_params() -> None:
    be, conn = _backend(pgvector_version="0.8.0")
    be.search(
        vector=_unit_query(),
        k=3,
        filters={"category": "books", "language": "en"},
    )
    search_calls = [(q, p) for q, p in conn.executed if isinstance(p, dict) and "query_vec" in p]
    assert search_calls
    sql, params = search_calls[-1]
    assert params["index_version"] == "idx-1"
    assert params["f_category"] == "books"
    assert params["f_language"] == "en"
    assert "f_brand" not in params
    assert "%(query_vec)s::vector" in sql


# --------------------------------------------------------------------------- #
# from_registry
# --------------------------------------------------------------------------- #
def test_from_registry_reads_active_index() -> None:
    conn = _FakeConnection(active_index_version="idx-42")
    be = PgvectorBackend.from_registry(conn, hnsw_ef_search=80)
    assert be.active_index_version == "idx-42"


def test_from_registry_fails_when_no_active_index() -> None:
    def no_active(sql: str) -> list[tuple[Any, ...]] | None:
        if "status = 'active'" in sql:
            return []
        return None

    with pytest.raises(RuntimeError, match="no active index"):
        PgvectorBackend.from_registry(_FakeConnection(responses=no_active), hnsw_ef_search=80)


# --------------------------------------------------------------------------- #
# defaults sanity
# --------------------------------------------------------------------------- #
def test_module_defaults_are_positive() -> None:
    assert DEFAULT_MAX_SCAN_TUPLES >= 1
    assert FALLBACK_OVERFETCH >= 2
