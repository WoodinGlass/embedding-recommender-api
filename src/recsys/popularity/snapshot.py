"""Tier 3 reader for the popularity snapshot (ADR-0020 § Tier 3).

The query is the one the ADR fixes: filter the snapshot by the
request's metadata fields, order by rank ascending, limit to k.
The filter fields come from ``FILTER_FIELDS`` (ADR-0008); the query
is parameterized; no field name is built from user input.

The score the response carries is a placeholder in `(0, 1]`:
``1.0 - rank / (max_rank + 1)``, so the first item's score is close
to 1.0 and every item scores above 0. The scale is the snapshot's,
not the ANN's; a caller that switches on ``meta.source`` should not
compare a fallback score to an ANN score.

The reader takes a connection, not a pool. The caller (the handler)
owns the pool and passes one connection for the duration of the
request. This matches ``PgvectorBackend`` (ADR-0012) and keeps the
reader a pure function of its arguments.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from recsys.retrieval.filters import FILTER_FIELDS

#: The default per-query timeout. The snapshot is a small table and
#: the query is bounded by rank; a query that exceeds this is a
#: signal, not a workload to wait on. The handler passes the
#: configured value.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 0.2


def _build_query(
    filters: Mapping[str, str] | None,
) -> tuple[str, list[str | None]]:
    """Return the SQL and the ordered filter values.

    The field order matches the values list so the caller passes
    ``params`` positionally and a missing field becomes a NULL
    comparison (which the WHERE clause treats as "no constraint").
    """
    filter_values: list[str | None] = []
    clauses: list[str] = []
    for field in FILTER_FIELDS:
        # The ``::text`` cast is required: without it, PostgreSQL's
        # extended-query protocol cannot infer the type of a parameter
        # used only in an ``IS NULL`` comparison, and psycopg raises
        # ``IndeterminateDatatype``. The column is TEXT.
        value = (filters or {}).get(field)
        clauses.append(f"({field} = %s::text OR %s::text IS NULL)")
        filter_values.append(value)
        filter_values.append(value)
    where = " AND ".join(clauses)
    sql = f"""
        WITH snapshot_size AS (
            SELECT COALESCE(MAX(rank), 0) AS n FROM popularity_snapshot
        )
        SELECT item_id, rank, (SELECT n FROM snapshot_size) AS n
        FROM popularity_snapshot
        WHERE {where}
        ORDER BY rank ASC
        LIMIT %s
    """  # noqa: S608 - the WHERE clause is built from FILTER_FIELDS constants
    return sql, filter_values


def read_snapshot(
    connection: Any,
    *,
    k: int,
    filters: Mapping[str, str] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[tuple[str, float]]:
    """Return up to ``k`` ``(item_id, score)`` pairs from the snapshot.

    Returns an empty list when the snapshot has no row matching the
    filters, which is the "no tier-3 answer" case that lets the
    chain fall through to tier 4 (ADR-0020 § What "fall through"
    means for the response). It does not raise for an empty result:
    the caller treats empty and missing the same way, and a query
    that returns nothing is a normal answer.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")

    timeout_ms = max(1, int(timeout_seconds * 1000))
    sql, filter_values = _build_query(filters)

    with connection.transaction(), connection.cursor() as cur:
        cur.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(timeout_ms),),
        )
        cur.execute(sql, (*filter_values, k))
        rows = cur.fetchall()

    result: list[tuple[str, float]] = []
    for item_id, rank, max_rank in rows:
        rank_i = int(rank)
        n = int(max_rank)
        score = 1.0 - rank_i / (n + 1)
        result.append((str(item_id), score))
    return result


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "read_snapshot",
]
