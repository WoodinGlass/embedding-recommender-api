"""Verify the sample catalog and golden set are consistent and complete.

These tests are the fast fixture version of the offline-evaluation gate
that lands in M2.5. They make no model calls; they assert that the input
data is well-formed and internally consistent, so any downstream
evaluation failure is attributable to the model, not to the fixture.

The golden set is now ``v1.jsonl`` (ADR-0009 § 1), loaded through
:func:`recsys.evaluation.golden_set.load_golden_set`. Using the same
loader that the evaluation harness uses means the test cannot pass with
a file the harness would reject.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

from recsys.evaluation.golden_set import GoldenSet, load_golden_set

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data" / "sample" / "catalog.jsonl"
GOLDEN_SET_PATH = REPO_ROOT / "evaluation" / "golden_set" / "v1.jsonl"

EXPECTED_ITEMS = 200
EXPECTED_TOPICS = 20
EXPECTED_ITEMS_PER_TOPIC = 10
EXPECTED_SEEDS = 3
EXPECTED_RELEVANT = 7


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #
def _load_catalog() -> list[dict[str, str]]:
    assert CATALOG_PATH.is_file(), f"missing {CATALOG_PATH}"
    lines = CATALOG_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_catalog_has_expected_size() -> None:
    assert len(_load_catalog()) == EXPECTED_ITEMS


def test_item_ids_are_unique_and_well_formed() -> None:
    import re

    ids = [it["item_id"] for it in _load_catalog()]
    assert len(set(ids)) == len(ids), "duplicate item_id"
    pattern = re.compile(r"^i_\d{4}$")
    for i in ids:
        assert pattern.match(i), f"malformed item_id: {i!r}"


def test_every_item_has_required_fields_and_no_empties() -> None:
    required = {"item_id", "title", "description", "category", "brand"}
    for it in _load_catalog():
        missing = required - set(it.keys())
        assert not missing, f"{it.get('item_id')}: missing {missing}"
        for k, v in it.items():
            assert isinstance(v, str), f"{it['item_id']}.{k} is not a string"
            assert v.strip(), f"{it['item_id']}.{k} is empty"


def test_categories_are_a_multiple_of_topic_size() -> None:
    # The invariant is 20 topics x 10 items. Categories are allowed to
    # span topics (books, instruments), so a category size is a multiple
    # of the topic size, never a fraction.
    by_cat: dict[str, list[str]] = {}
    for it in _load_catalog():
        by_cat.setdefault(it["category"], []).append(it["item_id"])
    for cat, ids in by_cat.items():
        assert len(ids) % EXPECTED_ITEMS_PER_TOPIC == 0, (
            f"{cat}: {len(ids)} items is not a multiple of {EXPECTED_ITEMS_PER_TOPIC}"
        )


# --------------------------------------------------------------------------- #
# golden set
# --------------------------------------------------------------------------- #
def _load_golden_set() -> GoldenSet:
    assert GOLDEN_SET_PATH.is_file(), f"missing {GOLDEN_SET_PATH}"
    # Load through the production loader with the catalog as known ids.
    known = {it["item_id"] for it in _load_catalog()}
    return load_golden_set(GOLDEN_SET_PATH, known_item_ids=known)


def test_golden_set_version_and_query_count() -> None:
    gs = _load_golden_set()
    assert gs.version == "v1"
    assert len(gs) == EXPECTED_TOPICS


def test_golden_set_each_query_shape() -> None:
    gs = _load_golden_set()
    for q in gs.queries:
        assert len(q.seed_item_ids) == EXPECTED_SEEDS, (
            f"{q.query_id}: {len(q.seed_item_ids)} seeds, expected {EXPECTED_SEEDS}"
        )
        assert len(q.relevant_item_ids) == EXPECTED_RELEVANT, (
            f"{q.query_id}: {len(q.relevant_item_ids)} relevant, expected {EXPECTED_RELEVANT}"
        )


def test_golden_set_covers_every_catalog_item() -> None:
    gs = _load_golden_set()
    catalog_ids = {it["item_id"] for it in _load_catalog()}
    covered: set[str] = set()
    for q in gs.queries:
        covered |= set(q.seed_item_ids)
        covered |= set(q.relevant_item_ids)
    assert covered == catalog_ids, "catalog items not covered by golden set"


def test_golden_set_seed_items_are_not_relevant_elsewhere() -> None:
    gs = _load_golden_set()
    all_seeds = {sid for q in gs.queries for sid in q.seed_item_ids}
    all_relevant = [rid for q in gs.queries for rid in q.relevant_item_ids]
    assert not (all_seeds & set(all_relevant)), (
        "an item is a seed in one query and relevant in another"
    )


def test_golden_set_topic_slugs_are_unique() -> None:
    gs = _load_golden_set()
    slugs = [q.topic for q in gs.queries]
    assert len(set(slugs)) == len(slugs), "duplicate topic slug in golden set"


# --------------------------------------------------------------------------- #
# generator determinism
# --------------------------------------------------------------------------- #
def test_generate_sample_catalog_is_deterministic() -> None:
    """The generator must produce identical bytes across two runs.

    This is the determinism contract for the fixture itself: the sample
    catalog is a pure function of ``scripts/sample_catalog_data.py``.
    """
    script = REPO_ROOT / "scripts" / "generate_sample_catalog.py"

    def run() -> tuple[str, str]:
        # `script` is derived from REPO_ROOT (resolved via pathlib) — it is
        # not user input. The noqa documents that the S603 warning was
        # considered and dismissed for this specific call site.
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
