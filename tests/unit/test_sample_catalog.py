"""Verify the sample catalog and golden set are consistent and complete.

These tests are the "fast fixture" version of the offline-evaluation gate
that lands in M2. They make no model calls; they only assert that the input
data is well-formed and internally consistent, so that any downstream
evaluation failure is attributable to the model, not to the fixture.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data" / "sample" / "catalog.jsonl"
GOLDEN_SET_PATH = REPO_ROOT / "evaluation" / "golden_set" / "queries.yaml"

ITEM_ID_RE = re.compile(r"^i_\d{4}$")
EXPECTED_ITEMS = 200
EXPECTED_TOPICS = 20
EXPECTED_ITEMS_PER_TOPIC = 10
EXPECTED_SEEDS = 3
EXPECTED_RELEVANT = 7


def _load_catalog() -> list[dict[str, str]]:
    assert CATALOG_PATH.is_file(), f"missing {CATALOG_PATH}"
    lines = CATALOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_golden_set() -> dict[str, Any]:
    assert GOLDEN_SET_PATH.is_file(), f"missing {GOLDEN_SET_PATH}"
    doc = yaml.safe_load(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def test_catalog_has_expected_size() -> None:
    assert len(_load_catalog()) == EXPECTED_ITEMS


def test_item_ids_are_unique_and_well_formed() -> None:
    ids = [it["item_id"] for it in _load_catalog()]
    assert len(set(ids)) == len(ids), "duplicate item_id"
    for i in ids:
        assert ITEM_ID_RE.match(i), f"malformed item_id: {i!r}"


def test_every_item_has_required_fields_and_no_empties() -> None:
    required = {"item_id", "title", "description", "category", "brand"}
    for it in _load_catalog():
        missing = required - set(it.keys())
        assert not missing, f"{it.get('item_id')}: missing {missing}"
        for k, v in it.items():
            assert isinstance(v, str), f"{it['item_id']}.{k} is not a string"
            assert v.strip(), f"{it['item_id']}.{k} is empty"


def test_twenty_topics_with_ten_items_each() -> None:
    """The catalog structure invariant is 20 topics × 10 items.

    Topic membership is defined by the golden set (seed + relevant), not by
    the ``category`` field. Categories are a broader taxonomy and are
    intentionally allowed to span topics: ``books`` covers both
    ``science_fiction`` and ``art_theory``; ``instruments`` covers both
    ``acoustic_guitars`` and ``synth_analog``. That overlap is a feature —
    filtering by ``category=books`` should return both topic clusters.
    """
    doc = _load_golden_set()
    for q in doc["queries"]:
        members = set(q["seed_item_ids"]) | set(q["relevant_item_ids"])
        assert len(members) == EXPECTED_ITEMS_PER_TOPIC, (
            f"{q['id']}: {len(members)} items, expected {EXPECTED_ITEMS_PER_TOPIC}"
        )


def test_categories_may_span_topics_but_are_bounded() -> None:
    """Categories are a many-topics-to-one-category relation.

    There are 20 topics and at most 20 categories. Fewer categories than
    topics is expected (books, instruments). More would indicate a typo in
    sample_catalog_data.py.
    """
    by_cat: dict[str, list[str]] = {}
    for it in _load_catalog():
        by_cat.setdefault(it["category"], []).append(it["item_id"])
    assert 1 < len(by_cat) <= EXPECTED_TOPICS, (
        f"got {len(by_cat)} categories; expected 2..{EXPECTED_TOPICS}"
    )
    for cat, ids in by_cat.items():
        assert len(ids) % EXPECTED_ITEMS_PER_TOPIC == 0, (
            f"{cat}: {len(ids)} items is not a multiple of "
            f"{EXPECTED_ITEMS_PER_TOPIC} — likely a data-entry mistake"
        )


def test_golden_set_header() -> None:
    doc = _load_golden_set()
    assert doc["version"] == 1
    assert doc["k_for_eval"] == 10
    assert doc["seed_count_per_query"] == EXPECTED_SEEDS
    assert len(doc["queries"]) == EXPECTED_TOPICS


def test_golden_set_seeds_relevant_disjoint_and_cover_catalog() -> None:
    catalog_ids = {it["item_id"] for it in _load_catalog()}
    doc = _load_golden_set()
    covered: set[str] = set()
    for q in doc["queries"]:
        seeds = set(q["seed_item_ids"])
        relevant = set(q["relevant_item_ids"])
        assert len(seeds) == EXPECTED_SEEDS, f"{q['id']}: expected {EXPECTED_SEEDS} seeds"
        assert len(relevant) == EXPECTED_RELEVANT, (
            f"{q['id']}: expected {EXPECTED_RELEVANT} relevant"
        )
        assert not (seeds & relevant), f"{q['id']}: seed and relevant overlap"
        assert (seeds | relevant) <= catalog_ids, f"{q['id']}: references unknown item_id"
        covered |= seeds | relevant
    assert covered == catalog_ids, "catalog items not covered by golden set"


def test_golden_set_seed_items_are_not_relevant_elsewhere() -> None:
    doc = _load_golden_set()
    all_seeds = {sid for q in doc["queries"] for sid in q["seed_item_ids"]}
    all_relevant = [rid for q in doc["queries"] for rid in q["relevant_item_ids"]]
    assert not (all_seeds & set(all_relevant)), (
        "an item is a seed in one query and relevant in another"
    )


def test_golden_set_topic_slugs_are_unique() -> None:
    doc = _load_golden_set()
    slugs = [q["topic"] for q in doc["queries"]]
    assert len(set(slugs)) == len(slugs), "duplicate topic slug in golden set"


def test_generate_sample_catalog_is_deterministic() -> None:
    """The generator must produce identical bytes across two runs.

    This is the determinism contract for the fixture itself: the sample
    catalog is a pure function of ``scripts/sample_catalog_data.py``.
    """
    import hashlib
    import subprocess
    import sys

    script = REPO_ROOT / "scripts" / "generate_sample_catalog.py"

    def run() -> tuple[str, str]:
        # `script` is derived from REPO_ROOT (resolved via pathlib) — it is not
        # user input. The noqa documents that the S603 warning was considered
        # and dismissed for this specific call site.
        subprocess.run(  # noqa: S603
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        catalog_hash = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
        golden_hash = hashlib.sha256(GOLDEN_SET_PATH.read_bytes()).hexdigest()
        return catalog_hash, golden_hash

    first = run()
    second = run()
    assert first == second, (
        f"generator is not deterministic:\n  first : {first}\n  second: {second}"
    )
