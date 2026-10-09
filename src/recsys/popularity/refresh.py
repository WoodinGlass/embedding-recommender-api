"""Refresh the popularity snapshot (ADR-0020 § Tier 3).

The snapshot is a small table (`popularity_snapshot`) holding a
pre-computed ranking, refreshed by a script an operator or a cron
runs. The refresh reads the top `size` items from the catalog and
replaces the table's contents inside one transaction: a concurrent
reader sees either the old snapshot or the new one, never an empty
table.

**The ranking is a placeholder in M3.** Without an event log (M5),
"popularity" cannot be measured; the order is `md5(item_id)`, which
is deterministic across runs and independent of insertion order,
but is not a measurement of anything real. The behavior the chain
depends on — a stable, filterable list of item ids — is what the
placeholder provides; the ranking's meaning arrives in M5. The
refresh function is the only place the placeholder lives; swapping
it for a real query is a change to one SQL statement.

Why not read the whole `item` table and sort in Python: at the
target catalog size (100k–1M) that is a full scan and a large sort
per refresh. The SQL query uses the index on `item_id` for the md5
ordering the same way a real popularity query would use an index on
a count column; the shape is what matters, not the placeholder's
semantics.
"""

from __future__ import annotations

from typing import Any, Final

#: The default statement timeout for the refresh. The refresh reads
#: a bounded number of rows and replaces a small table; a query that
#: exceeds this is a signal worth investigating, not a workload to
#: wait on. The script exposes it as a flag for a slower environment.
DEFAULT_REFRESH_TIMEOUT_SECONDS: Final[float] = 10.0


def refresh_popularity_snapshot(
    connection: Any,
    *,
    size: int,
    timeout_seconds: float = DEFAULT_REFRESH_TIMEOUT_SECONDS,
) -> int:
    """Replace the popularity snapshot with the top ``size`` items.

    Returns the number of rows written. Raises on a database error;
    the caller (the script) turns that into an exit code and a JSON
    line.

    The whole refresh runs in one transaction: a reader sees the
    old snapshot until the commit and the new one after. A partial
    refresh (an empty table for the duration of the insert) would
    make tier 3 fall through to tier 4 for no reason, which is the
    opposite of what the fallback chain is for.

    The timeout is set with ``set_config(..., is_local=true)``: the
    ``SET LOCAL ... = $1`` form is rejected by PostgreSQL's
    extended-query protocol (see ``PgRecencyProvider`` for the same
    fix in M3.4/M3.5).
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size!r}")
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")

    timeout_ms = max(1, int(timeout_seconds * 1000))

    with connection.transaction(), connection.cursor() as cur:
        cur.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(timeout_ms),),
        )
        cur.execute(
            """
                SELECT item_id, category, brand, language
                FROM item
                ORDER BY md5(item_id)
                LIMIT %s
                """,
            (size,),
        )
        rows = cur.fetchall()

        cur.execute("DELETE FROM popularity_snapshot")
        if rows:
            cur.executemany(
                """
                    INSERT INTO popularity_snapshot (
                        item_id, rank, category, brand, language, refreshed_at
                    )
                    VALUES (%s, %s, %s, %s, %s, now())
                    """,
                [(row[0], i + 1, row[1], row[2], row[3]) for i, row in enumerate(rows)],
            )
    return len(rows)


__all__ = [
    "DEFAULT_REFRESH_TIMEOUT_SECONDS",
    "refresh_popularity_snapshot",
]
