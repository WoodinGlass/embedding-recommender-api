"""Generate data/sample/catalog.jsonl and evaluation/golden_set/v1.jsonl.

Deterministic: the same input (``sample_catalog_data.py``) always produces
byte-identical output. Re-run after any change to that file.

Item IDs are assigned sequentially in topic order: ``i_0001`` through
``i_0200``. IDs are opaque per ``docs/contracts.md`` § 0; nothing in the
system parses them.

Golden set: for each topic, the first three items are seeds and the
remaining seven are relevant. The format is fixed by
``docs/adr/0009-golden-set-and-metrics.md`` § 1 — one JSON object per
line, four required fields, no extras. The version is the filename
(``v1.jsonl``); there is no version field inside the file.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from sample_catalog_data import TOPICS  # noqa: E402

CATALOG_PATH = REPO_ROOT / "data" / "sample" / "catalog.jsonl"
GOLDEN_SET_DIR = REPO_ROOT / "evaluation" / "golden_set"
GOLDEN_SET_PATH = GOLDEN_SET_DIR / "v1.jsonl"

SEED_COUNT = 3


def generate_catalog() -> list[dict[str, str]]:
    """Return the catalog as a list of dicts, in deterministic order."""
    items: list[dict[str, str]] = []
    counter = 1
    for topic in TOPICS:
        brands = topic["brands"]
        raw_items = topic["items"]
        category = topic["category"]
        for i, entry in enumerate(raw_items):
            title, description = entry
            items.append(
                {
                    "item_id": f"i_{counter:04d}",
                    "title": title,
                    "description": description,
                    "category": category,
                    "brand": brands[i % len(brands)],
                }
            )
            counter += 1
    return items


def write_catalog(items: list[dict[str, str]]) -> None:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    CATALOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_golden_queries(items: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return the golden set queries, one per topic.

    Each query has exactly the four fields required by ADR-0009 § 1:
    ``query_id``, ``topic``, ``seed_item_ids``, ``relevant_item_ids``.
    Extra fields are rejected by the loader, so this generator must not
    add any.
    """
    queries: list[dict[str, Any]] = []
    idx = 0
    for topic in TOPICS:
        topic_size = len(topic["items"])
        ids = [it["item_id"] for it in items[idx : idx + topic_size]]
        idx += topic_size
        queries.append(
            {
                "query_id": f"q_{topic['slug']}",
                "topic": topic["slug"],
                "seed_item_ids": ids[:SEED_COUNT],
                "relevant_item_ids": ids[SEED_COUNT:],
            }
        )
    return queries


def write_golden_set(queries: list[dict[str, Any]]) -> None:
    """Write the golden set as JSONL: one query per line, no trailing blank.

    Keys are emitted in a stable order so the file is byte-identical
    across runs. ``sort_keys=True`` inside each line, joined with ``\\n``
    and a final newline.
    """
    GOLDEN_SET_DIR.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(q, sort_keys=True, ensure_ascii=False) for q in queries]
    GOLDEN_SET_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    items = generate_catalog()
    write_catalog(items)
    queries = generate_golden_queries(items)
    write_golden_set(queries)

    summary = {
        "event": "sample_catalog.generated",
        "items": len(items),
        "topics": len(TOPICS),
        "queries": len(queries),
        "catalog_path": str(CATALOG_PATH.relative_to(REPO_ROOT)),
        "golden_set_path": str(GOLDEN_SET_PATH.relative_to(REPO_ROOT)),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
