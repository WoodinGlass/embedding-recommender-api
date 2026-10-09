"""Popularity snapshot (ADR-0020 § Tier 3).

The pre-computed ranking tier 3 reads when the ANN path fails. One
row per item in the snapshot; the rank is a small integer starting
at 1. The filter columns (`category`, `brand`, `language`) are
copied from `item` so the tier-3 query can apply the request's
filters in SQL, without a join at request time.

The snapshot is refreshed from the item table by
`scripts/refresh_popularity.py`. In M3 the ranking is a placeholder:
without an event log (M5), "popularity" is a deterministic order by
`md5(item_id)` — stable across runs and independent of insertion
order, but not a measurement of anything real. The placeholder is
documented at the refresh function; the table's shape does not
change when M5 swaps the ranking for an event count.

The `ON DELETE CASCADE` on `item_id` means the snapshot stays
consistent if a catalog item is removed; the next refresh would
drop the row anyway, but the cascade covers the window between.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-09
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE popularity_snapshot (
            item_id      TEXT PRIMARY KEY
                         REFERENCES item (item_id) ON DELETE CASCADE,
            rank         INTEGER NOT NULL CHECK (rank >= 1),
            category     TEXT NOT NULL,
            brand        TEXT NOT NULL,
            language     TEXT NOT NULL,
            refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # Tier 3 orders by rank ASC and limits by k. The rank index
    # makes the LIMIT a bounded scan instead of a sort.
    op.execute(
        """
        CREATE INDEX popularity_snapshot_rank_idx
            ON popularity_snapshot (rank)
        """
    )
    # The filter columns are the tier-3 WHERE clause; a composite
    # index is not built because the selectivity of each filter
    # varies and the planner picks the rank index in practice. A
    # future measurement can add one; the table is small.


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS popularity_snapshot_rank_idx")
    op.execute("DROP TABLE IF EXISTS popularity_snapshot")
