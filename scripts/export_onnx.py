#!/usr/bin/env python3
"""CLI wrapper for :func:`recsys.embeddings.onnx_export.export`.

    python scripts/export_onnx.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --out artifacts/onnx/sentence-transformers__all-MiniLM-L6-v2

Prints a single JSON line to stdout. Exit codes:

    0 — success or skipped (artifact already present)
    2 — input error (bad path, missing model name)
    3 — export error (missing extra, unsupported model)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from recsys.embeddings.onnx_export import export  # noqa: E402


def model_slug(name: str) -> str:
    """Turn a model name into a filesystem-safe slug.

    ``sentence-transformers/all-MiniLM-L6-v2`` →
    ``sentence-transformers__all-MiniLM-L6-v2``.
    """
    return name.replace("/", "__")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="sentence-transformers model name",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="output directory (default: artifacts/onnx/<model_slug>)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-export even if the artifact already exists",
    )
    args = parser.parse_args(argv)

    out = args.out or (REPO_ROOT / "artifacts" / "onnx" / model_slug(args.model))

    try:
        summary = export(args.model, out, force=args.force)
    except RuntimeError as exc:
        print(json.dumps({"event": "export.error", "message": str(exc)}))
        return 3

    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
