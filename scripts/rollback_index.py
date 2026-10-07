#!/usr/bin/env python3
"""Revert to the most recently retired index.

    python scripts/rollback_index.py

This is a pointer swap: no embeddings are moved, no rebuild is performed.
Exit codes match promote_index.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys.retrieval.promote import (  # noqa: E402
    PromoteError,
    rollback,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_CONNECTION_ERROR = 3
EXIT_LOCK_TIMEOUT = 4


def _open_connection() -> Any:
    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError("psycopg is not installed. Install with: pip install -e '.[db]'") from e
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    return psycopg.connect(url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    try:
        connection = _open_connection()
    except Exception as e:
        print(json.dumps({"event": "index.rollback.connection_error", "message": str(e)}))
        return EXIT_CONNECTION_ERROR

    try:
        result = rollback(connection, timeout_s=args.lock_timeout)
    except PromoteError as e:
        message = str(e)
        if "could not acquire promote lock" in message:
            print(json.dumps({"event": "index.rollback.lock_timeout", "message": message}))
            return EXIT_LOCK_TIMEOUT
        print(json.dumps({"event": "index.rollback.input_error", "message": message}))
        return EXIT_INPUT_ERROR
    except Exception as e:
        print(json.dumps({"event": "index.rollback.error", "message": str(e)}))
        return EXIT_INPUT_ERROR
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "event": "index.rollback",
                "status": result.status,
                "index_version": result.index_version,
                "previous_index_version": result.previous_index_version,
                "duration_ms": result.duration_ms,
            }
        )
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
