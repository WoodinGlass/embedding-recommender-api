"""Initial pgvector schema: item, embedding, index_registry.

Implements the schema defined in docs/adr/0006-pgvector-schema.md. The
dimension of the vector column is fixed at 384, matching the model pinned
in EMBEDDING_MODEL (sentence-transformers/all-MiniLM-L6-v2). Changing the
dimension requires a new migration and an ADR; pgvector does not support
ALTER COLUMN ... TYPE vector(M).

Revision ID: 0001
Revises:
Create Date: 2026-10-08
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # Extension
    # ------------------------------------------------------------------ #
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------ #
    # item — catalog rows and filterable metadata
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE TABLE item (
            item_id       TEXT PRIMARY KEY,
            title         TEXT NOT NULL,
            description   TEXT NOT NULL,
            category      TEXT NOT NULL,
            brand         TEXT NOT NULL,
            language      TEXT NOT NULL DEFAULT 'en',
            content_hash  TEXT NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX item_category_idx ON item (category)")
    op.execute("CREATE INDEX item_brand_idx    ON item (brand)")
    op.execute("CREATE INDEX item_language_idx ON item (language)")

    # ------------------------------------------------------------------ #
    # index_registry — the source of truth for what each index is
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE TABLE index_registry (
            index_version          TEXT PRIMARY KEY,
            model_version          TEXT NOT NULL,
            catalog_snapshot       TEXT NOT NULL,
            preprocessing_version  TEXT NOT NULL,
            metric                 TEXT NOT NULL
                                   CHECK (metric IN ('cosine', 'l2', 'ip')),
            hnsw_m                 INTEGER NOT NULL CHECK (hnsw_m >= 2),
            hnsw_ef_construction   INTEGER NOT NULL
                                   CHECK (hnsw_ef_construction >= 2),
            hnsw_ef_search         INTEGER NOT NULL CHECK (hnsw_ef_search >= 1),
            pgvector_version       TEXT NOT NULL,
            golden_set_version     TEXT NOT NULL,
            row_count              INTEGER NOT NULL DEFAULT 0,
            status                 TEXT NOT NULL
                                   CHECK (status IN ('building', 'active', 'retired')),
            created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
            activated_at           TIMESTAMPTZ,
            retired_at             TIMESTAMPTZ
        )
        """
    )

    # At most one active index at any moment.
    op.execute(
        """
        CREATE UNIQUE INDEX index_registry_one_active
            ON index_registry (status) WHERE status = 'active'
        """
    )

    # ------------------------------------------------------------------ #
    # embedding — one row per (item_id, index_version)
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE TABLE embedding (
            item_id         TEXT NOT NULL REFERENCES item (item_id)
                            ON DELETE CASCADE,
            index_version   TEXT NOT NULL REFERENCES index_registry (index_version)
                            ON DELETE CASCADE,
            model_version   TEXT NOT NULL,
            vector          vector(384) NOT NULL,
            PRIMARY KEY (item_id, index_version)
        )
        """
    )

    # HNSW index for cosine distance. Build parameters come from config at
    # build time; the values below are the same defaults as
    # HNSW_M / HNSW_EF_CONSTRUCTION in .env.example. A migration is not the
    # place to change them — the build script does, with an ADR.
    op.execute(
        """
        CREATE INDEX embedding_hnsw_cosine_idx
            ON embedding USING hnsw (vector vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    # Drop in reverse dependency order. The vector extension is left in
    # place: dropping it would break other databases on the same cluster
    # and is not this migration's responsibility.
    op.execute("DROP INDEX IF EXISTS embedding_hnsw_cosine_idx")
    op.execute("DROP TABLE IF EXISTS embedding")
    op.execute("DROP INDEX IF EXISTS index_registry_one_active")
    op.execute("DROP TABLE IF EXISTS index_registry")
    op.execute("DROP INDEX IF EXISTS item_language_idx")
    op.execute("DROP INDEX IF EXISTS item_brand_idx")
    op.execute("DROP INDEX IF EXISTS item_category_idx")
    op.execute("DROP TABLE IF EXISTS item")
