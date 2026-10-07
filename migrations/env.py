"""Alembic environment.

Reads DATABASE_URL from the environment (never from alembic.ini), so a
migration run in a different environment does not require editing a
checked-in file. The URL is required: if it is not set, this module exits
with a clear error rather than silently operating on a default database.
"""

from __future__ import annotations

import os
import pathlib
import sys

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make src/ importable so that migration scripts can import from recsys if
# they ever need to (they should not need to; raw SQL is preferred).
_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Alembic Config object, giving access to the values in alembic.ini.
config = context.config

# The URL is required. No fallback, no default.
url = os.environ.get("DATABASE_URL")
if not url:
    raise SystemExit(
        "DATABASE_URL is not set. Export it before running alembic, e.g.\n"
        "    export DATABASE_URL='postgresql://recsys:recsys@localhost:5432/recsys'"
    )
config.set_main_option("sqlalchemy.url", url)

# Metadata for autogenerate. Migrations in this project are written by hand
# (raw SQL), so autogenerate is not used; the metadata object is empty.
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations without a live connection (emit SQL to stdout)."""
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            transactional_ddl=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
