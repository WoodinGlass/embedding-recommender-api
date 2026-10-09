"""Experiment exposure log (ADR-0017 § Exposure is logged).

One row per (experiment, user_id, request) that actually served a
variant. Distinct from the `event` table (M3.6) because the two
carry different fields: an exposure has `fallback_reason` and a
`paused` flag that a click or an impression does not, and a click
has an `item_id` that an exposure does not. ADR-0017 says "same
event store" — the meaning is the same PostgreSQL instance and the
same retention policy, not one table for two shapes.

The primary key is the exposure's `event_id`, derived by the writer
as `sha256(f"exposure:{experiment}:{request_id}")`. The uniqueness
is what makes a retried request idempotent (ADR-0017 § Exposure is
idempotent by `request_id`): the second insert conflicts and is
ignored.

Indices support the analyses the M5 and M7 readers run:

- `(experiment, event_ts)` — time-bounded scans per experiment,
  which is how SRM and significance are computed;
- `(experiment, variant)` — the SRM denominator count;
- `(user_id_hash, experiment)` — grouping an experiment's exposures
  by user, which the churn features (M7) also read;
- `request_id` — correlating an exposure with the recommend
  response that produced it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-09
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE experiment_exposure (
            event_id                TEXT PRIMARY KEY,
            experiment              TEXT NOT NULL,
            variant                 TEXT NOT NULL,
            request_id              TEXT NOT NULL,
            user_id_hash            TEXT NOT NULL
                                    CHECK (user_id_hash ~ '^[0-9a-f]{64}$'),
            user_id_hash_version    INTEGER NOT NULL
                                    CHECK (user_id_hash_version >= 0),
            bucket                  INTEGER NOT NULL
                                    CHECK (bucket >= 0 AND bucket < 10000),
            effective_salt          TEXT NOT NULL,
            paused                  BOOLEAN NOT NULL DEFAULT FALSE,
            fallback_reason         TEXT
                                    CHECK (
                                        fallback_reason IS NULL
                                        OR fallback_reason IN (
                                            'paused',
                                            'stopped',
                                            'disabled',
                                            'allocation_gap'
                                        )
                                    ),
            event_ts                TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE INDEX experiment_exposure_experiment_ts_idx
            ON experiment_exposure (experiment, event_ts)
        """
    )
    op.execute(
        """
        CREATE INDEX experiment_exposure_experiment_variant_idx
            ON experiment_exposure (experiment, variant)
        """
    )
    op.execute(
        """
        CREATE INDEX experiment_exposure_user_experiment_idx
            ON experiment_exposure (user_id_hash, experiment)
        """
    )
    op.execute(
        """
        CREATE INDEX experiment_exposure_request_id_idx
            ON experiment_exposure (request_id)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS experiment_exposure_request_id_idx")
    op.execute("DROP INDEX IF EXISTS experiment_exposure_user_experiment_idx")
    op.execute("DROP INDEX IF EXISTS experiment_exposure_experiment_variant_idx")
    op.execute("DROP INDEX IF EXISTS experiment_exposure_experiment_ts_idx")
    op.execute("DROP TABLE IF EXISTS experiment_exposure")
