#!/usr/bin/env python3
"""Build an index from the active embedding run.

    python scripts/build_index.py
    python scripts/build_index.py --run-id 2026-10-08T11-30-00Z__minilm-onnx-v1+a3f9e021
    python scripts/build_index.py --promote       # build then activate

Prints a single JSON line to stdout. Exit codes:

    0 — success or duplicate (idempotent)
    2 — input error (missing run, missing catalog, malformed manifest)
    3 — connection error (psycopg missing, DATABASE_URL missing, DB down)
    4 — build error (constraint violation, pk collision beyond ext)
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

from recsys.retrieval.build import (  # noqa: E402
    BuildInputError,
    build_index,
    read_catalog_items,
    resolve_build_inputs,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_CONNECTION_ERROR = 3
EXIT_BUILD_ERROR = 4


def _open_connection() -> Any:
    """Open a psycopg connection from DATABASE_URL. Raises on failure."""
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
        "--embeddings-root",
        type=pathlib.Path,
        default=REPO_ROOT / "artifacts" / "embeddings",
        help="root of the embedding artifact tree (contains current and runs/)",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="build a specific run instead of the one named by current",
    )
    p.add_argument(
        "--catalog",
        type=pathlib.Path,
        default=REPO_ROOT / "data" / "sample" / "catalog.jsonl",
        help="JSONL catalog file used to populate the item table",
    )
    p.add_argument("--metric", default="cosine", choices=("cosine", "l2", "ip"))
    p.add_argument("--hnsw-m", type=int, default=None)
    p.add_argument("--hnsw-ef-construction", type=int, default=None)
    p.add_argument("--hnsw-ef-search", type=int, default=None)
    p.add_argument(
        "--golden-set-version",
        default="v1",
        help="golden set the eventual thresholds apply to (ADR-0009)",
    )
    p.add_argument("--batch-size", type=int, default=1_000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)

    # Config defaults from settings when not overridden.
    from recsys.config.settings import get_settings

    settings = get_settings()
    hnsw_m = args.hnsw_m if args.hnsw_m is not None else settings.hnsw_m
    hnsw_ef_construction = (
        args.hnsw_ef_construction
        if args.hnsw_ef_construction is not None
        else settings.hnsw_ef_construction
    )
    hnsw_ef_search = (
        args.hnsw_ef_search if args.hnsw_ef_search is not None else settings.hnsw_ef_search
    )

    try:
        inputs = resolve_build_inputs(args.embeddings_root, run_id=args.run_id)
        catalog_items = read_catalog_items(args.catalog)
    except BuildInputError as e:
        print(json.dumps({"event": "index.build.input_error", "message": str(e)}))
        return EXIT_INPUT_ERROR

    try:
        connection = _open_connection()
    except Exception as e:
        print(json.dumps({"event": "index.build.connection_error", "message": str(e)}))
        return EXIT_CONNECTION_ERROR

    try:
        result = build_index(
            connection,
            inputs=inputs,
            catalog_items=catalog_items,
            metric=args.metric,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
            golden_set_version=args.golden_set_version,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(json.dumps({"event": "index.build.error", "message": str(e)}))
        return EXIT_BUILD_ERROR
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "event": "index.build",
                "status": result.status,
                "index_version": result.index_version,
                "row_count": result.row_count,
                "duration_ms": result.duration_ms,
                "run_id": inputs.run_id,
                "model_version": inputs.model_version,
                "catalog_snapshot": inputs.catalog_snapshot,
            }
        )
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
