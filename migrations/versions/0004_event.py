"""Event log (ADR-0018 § Decision).

One row per interaction event a client sent to `POST /v1/events`. The
table is the same PostgreSQL instance the exposure log lives in
(ADR-0017 § "The same event store" — the same instance, not one
table); the two are separate because their shapes differ: an event
has `item_id` and `position` that an exposure does not, an exposure
has `fallback_reason` and `paused` that an event does not.

**`event_id` is the primary key.** A client-supplied id, or one the
server generates from `sha256(f"{request_id}:{index}")` when the
client omitted it. The uniqueness is what makes a retry of a batch
idempotent: the second insert conflicts and is ignored (ADR-0018 §
Idempotency: client-supplied `event_id`).

**No foreign key on `item_id`.** An event is a historical record; the
item it refers to may be removed from the catalog later, and the
event is still valid data. Referential integrity is the event's
payload, not the current catalog state. The alternative (a cascading
delete that removes events when an item is dropped) would delete
rows the analysis needs.

**PII.** The raw `user_id` is never stored; the row carries
`user_id_hash` (HMAC-SHA256, 64 lowercase hex) and
`user_id_hash_version` so a salt rotation is a filter, not a data
loss (ADR-0018 § User id hashing).

**Retention.** `EVENT_RETENTION_DAYS` (config, default 365) is the
window after which the M6 scheduled job deletes rows. The index on
`event_ts` is what the job scans; it is not the API path's index.

Indices support the two readers:

- `(user_id_hash, event_ts)` — the churn features (M7) group by user
  and time.
- `(item_id, event_ts)` — item-level analysis (M5).
- `(event_ts)` — the retention job.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-09
"""

from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE event (
            event_id              TEXT PRIMARY KEY,
            event_ts              TIMESTAMPTZ NOT NULL,
            ingested_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
            event_type            TEXT NOT NULL
                                  CHECK (event_type IN ('impression', 'click', 'conversion')),
            user_id_hash          TEXT NOT NULL
                                  CHECK (user_id_hash ~ '^[0-9a-f]{64}$'),
            user_id_hash_version  INTEGER NOT NULL
                                  CHECK (user_id_hash_version >= 0),
            item_id               TEXT NOT NULL,
            request_id            TEXT,
            experiment_name       TEXT,
            experiment_variant    TEXT,
            position              INTEGER
                                  CHECK (position IS NULL OR position >= 1),
            value                 DOUBLE PRECISION,
            CHECK ((experiment_name IS NULL) = (experiment_variant IS NULL))
        )
        """
    )
    op.execute(
        """
        CREATE INDEX event_ts_idx
            ON event (event_ts)
        """
    )
    op.execute(
        """
        CREATE INDEX event_user_hash_ts_idx
            ON event (user_id_hash, event_ts)
        """
    )
    op.execute(
        """
        CREATE INDEX event_item_ts_idx
            ON event (item_id, event_ts)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS event_item_ts_idx")
    op.execute("DROP INDEX IF EXISTS event_user_hash_ts_idx")
    op.execute("DROP INDEX IF EXISTS event_ts_idx")
    op.execute("DROP TABLE IF EXISTS event")
