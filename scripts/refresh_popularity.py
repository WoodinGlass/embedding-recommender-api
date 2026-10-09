#!/usr/bin/env python3
"""Refresh the popularity snapshot from the item table.

    python scripts/refresh_popularity.py
    python scripts/refresh_popularity.py --size 500

Prints one JSON line to stdout. Exit codes:

    0 — success
    2 — input error (bad --size)
    3 — connection error (psycopg missing, DATABASE_URL missing, DB down)
    4 — refresh error (a database error during the transaction)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys.popularity import (  # noqa: E402
    fetch_snapshot_rows,
    refresh_popularity_snapshot,
    write_popularity_file,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_CONNECTION_ERROR = 3
EXIT_REFRESH_ERROR = 4

DEFAULT_SIZE = 1_000


def _open_connection() -> Any:
    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError("psycopg is not installed. Install with: pip install -e '.[db]'") from e
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Export it, e.g.\n"
            "    export DATABASE_URL='postgresql://recsys:recsys@localhost:5432/recsys'"
        )
    return psycopg.connect(url)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="number of items in the snapshot (default: %(default)s)",
    )
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="statement timeout for the refresh (default: %(default)s)",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "popularity" / "snapshot.json",
        help=(
            "path to write the tier-4 cache file (default: %(default)s); "
            "the file is read at process startup (ADR-0020 § Tier 4)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    if args.size < 1:
        print(
            json.dumps(
                {
                    "event": "popularity.refresh.input_error",
                    "message": f"--size must be >= 1, got {args.size}",
                }
            )
        )
        return EXIT_INPUT_ERROR

    try:
        connection = _open_connection()
    except RuntimeError as e:
        print(json.dumps({"event": "popularity.refresh.connection_error", "message": str(e)}))
        return EXIT_CONNECTION_ERROR

    try:
        with connection:
            rows = refresh_popularity_snapshot(
                connection,
                size=args.size,
                timeout_seconds=args.timeout_seconds,
            )
            snapshot_rows = fetch_snapshot_rows(connection, size=args.size)
    except Exception as e:
        print(
            json.dumps({"event": "popularity.refresh.error", "message": f"{type(e).__name__}: {e}"})
        )
        return EXIT_REFRESH_ERROR
    finally:
        # Best-effort close. A connection that already broke is the
        # OS's to clean up; a close that raises does not change the
        # exit code the script is about to return.
        with contextlib.suppress(Exception):
            connection.close()

    try:
        write_popularity_file(path=args.out, items=snapshot_rows)
    except Exception as e:
        print(
            json.dumps(
                {
                    "event": "popularity.refresh.file_error",
                    "message": f"{type(e).__name__}: {e}",
                    "path": str(args.out),
                }
            )
        )
        return EXIT_REFRESH_ERROR

    print(
        json.dumps(
            {
                "event": "popularity.refresh.done",
                "rows": rows,
                "file_rows": len(snapshot_rows),
                "file_path": str(args.out),
                "size_requested": args.size,
            }
        )
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
