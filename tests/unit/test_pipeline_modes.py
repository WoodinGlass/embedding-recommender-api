"""Unit tests for the pipeline's catalog loading and mode planning.

No encoder, no ONNX. The heavy end-to-end path lives in
``tests/integration/test_pipeline_end_to_end.py``. These tests pin the two
behaviours that decide "how much work does a run do?":

- :func:`load_catalog` — strict input validation.
- :func:`plan_batch` / :func:`plan_incremental` — which items need encoding,
  and when a run is a no-op.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from recsys.embeddings import artifacts as art
from recsys.embeddings.pipeline import (
    InputError,
    encode_lazily,
    load_catalog,
    plan_batch,
    plan_incremental,
)
from recsys.embeddings.preprocess import CatalogItem, item_content_hash

pytestmark = pytest.mark.unit


# ============================================================================
# helpers
# ============================================================================
def _items() -> list[CatalogItem]:
    return [
        {
            "item_id": "i_0002",
            "title": "Hyperion",
            "description": "Seven pilgrims.",
            "category": "books",
            "brand": "tor",
        },
        {
            "item_id": "i_0001",
            "title": "Dune",
            "description": "A desert planet epic.",
            "category": "books",
            "brand": "ace",
        },
        {
            "item_id": "i_0003",
            "title": "Foundation",
            "description": "Statistical science.",
            "category": "books",
            "brand": "orbit",
        },
    ]


def _state_doc(
    *,
    items: dict[str, str],
    catalog_snapshot: str = "sha256:0000000000000000",
    model_version: str = "m+v1",
    preprocessing_version: str = "v1",
    config_hash: str = "sha256:deadbeef",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_version": model_version,
        "preprocessing_version": preprocessing_version,
        "config_hash": config_hash,
        "hash_algorithm": "sha256",
        "catalog_snapshot": catalog_snapshot,
        "items": items,
    }


def _valid_state_read(doc: dict[str, object]) -> art.StateRead:
    return art.StateRead(state=doc, reason=None)


# ============================================================================
# load_catalog
# ============================================================================
def _write_catalog(path: pathlib.Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_catalog_ok(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_catalog(
        p,
        [
            '{"item_id":"i_0001","title":"A","description":"d","category":"c","brand":"b"}',
            '{"item_id":"i_0002","title":"B","description":"d","category":"c","brand":"b"}',
        ],
    )
    items = load_catalog(p)
    assert [it["item_id"] for it in items] == ["i_0001", "i_0002"]


def test_load_catalog_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(InputError, match="not found"):
        load_catalog(tmp_path / "nope.jsonl")


def test_load_catalog_malformed_json(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(InputError, match="invalid JSON"):
        load_catalog(p)


def test_load_catalog_not_object(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(InputError, match="not a JSON object"):
        load_catalog(p)


def test_load_catalog_missing_field(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text('{"item_id":"i_0001","title":"A"}\n', encoding="utf-8")
    with pytest.raises(InputError, match="missing fields"):
        load_catalog(p)


def test_load_catalog_empty_item_id(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text(
        '{"item_id":"","title":"A","description":"d","category":"c","brand":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="non-empty string"):
        load_catalog(p)


def test_load_catalog_duplicate_id(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    line = '{"item_id":"i_0001","title":"A","description":"d","category":"c","brand":"b"}'
    _write_catalog(p, [line, line])
    with pytest.raises(InputError, match="duplicate item_id"):
        load_catalog(p)


def test_load_catalog_empty_file(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(InputError, match="no items"):
        load_catalog(p)


def test_load_catalog_skips_blank_lines(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.jsonl"
    _write_catalog(
        p,
        [
            "",
            '{"item_id":"i_0001","title":"A","description":"d","category":"c","brand":"b"}',
            "   ",
        ],
    )
    assert len(load_catalog(p)) == 1


# ============================================================================
# plan_batch
# ============================================================================
def test_plan_batch_encodes_everything_sorted() -> None:
    items = _items()
    plan = plan_batch(items)
    assert [it["item_id"] for it in plan.sorted_items] == [
        "i_0001",
        "i_0002",
        "i_0003",
    ]
    assert [it["item_id"] for it in plan.items_to_encode] == [
        "i_0001",
        "i_0002",
        "i_0003",
    ]
    assert plan.unchanged_ids == frozenset()
    assert plan.mode == "batch"
    assert plan.no_changes is False
    assert plan.state_reason is None


def test_plan_batch_content_hashes_match_item_hash() -> None:
    items = _items()
    plan = plan_batch(items)
    for it in items:
        assert plan.content_hashes[it["item_id"]] == item_content_hash(it)


# ============================================================================
# plan_incremental
# ============================================================================
def test_plan_incremental_no_state_encodes_all() -> None:
    items = _items()
    plan = plan_incremental(
        items,
        state_read=art.StateRead(state=None, reason="missing"),
    )
    assert len(plan.items_to_encode) == 3
    assert plan.unchanged_ids == frozenset()
    assert plan.mode == "incremental"
    assert plan.no_changes is False
    assert plan.state_reason == "missing"


def test_plan_incremental_all_unchanged_is_noop() -> None:
    items = _items()
    # Compute the actual hashes and snapshot, so nothing appears changed.
    from recsys.embeddings.preprocess import catalog_snapshot

    sorted_items = sorted(items, key=lambda it: it["item_id"])
    snap = catalog_snapshot(sorted_items)
    items_map = {it["item_id"]: item_content_hash(it) for it in sorted_items}
    doc = _state_doc(items=items_map, catalog_snapshot=snap)

    plan = plan_incremental(items, state_read=_valid_state_read(doc))
    assert plan.items_to_encode == []
    assert plan.unchanged_ids == frozenset({"i_0001", "i_0002", "i_0003"})
    assert plan.no_changes is True


def test_plan_incremental_partial_change() -> None:
    items = _items()
    from recsys.embeddings.preprocess import catalog_snapshot

    sorted_items = sorted(items, key=lambda it: it["item_id"])
    snap = catalog_snapshot(sorted_items)
    items_map = {it["item_id"]: item_content_hash(it) for it in sorted_items}
    # Pretend i_0002 has drifted: replace its recorded hash.
    items_map["i_0002"] = "stalehash0000000"
    doc = _state_doc(items=items_map, catalog_snapshot=snap)

    plan = plan_incremental(items, state_read=_valid_state_read(doc))
    assert [it["item_id"] for it in plan.items_to_encode] == ["i_0002"]
    assert plan.unchanged_ids == frozenset({"i_0001", "i_0003"})
    assert plan.no_changes is False


def test_plan_incremental_new_item_is_encoded() -> None:
    items = _items()
    sorted_items = sorted(items, key=lambda it: it["item_id"])
    # State contains only i_0001 and i_0002; i_0003 is new.
    items_map = {
        "i_0001": item_content_hash(sorted_items[0]),
        "i_0002": item_content_hash(sorted_items[1]),
    }
    doc = _state_doc(items=items_map, catalog_snapshot="sha256:othersnapshot00")

    plan = plan_incremental(items, state_read=_valid_state_read(doc))
    assert [it["item_id"] for it in plan.items_to_encode] == ["i_0003"]
    assert plan.unchanged_ids == frozenset({"i_0001", "i_0002"})
    # Snapshot differs from the state's snapshot, so not a no-op.
    assert plan.no_changes is False


def test_plan_incremental_deleted_item_is_dropped() -> None:
    # Previous state has an item no longer in the catalog.
    items = _items()[:2]
    sorted_items = sorted(items, key=lambda it: it["item_id"])
    items_map = {it["item_id"]: item_content_hash(it) for it in sorted_items}
    items_map["i_9999"] = "orphanhash000000"
    doc = _state_doc(items=items_map, catalog_snapshot="sha256:othersnapshot00")

    plan = plan_incremental(items, state_read=_valid_state_read(doc))
    assert plan.items_to_encode == []
    assert plan.unchanged_ids == frozenset({"i_0001", "i_0002"})
    # Snapshot differs because the catalog set changed.
    assert plan.no_changes is False


def test_plan_incremental_snapshot_mismatch_prevents_noop() -> None:
    items = _items()
    # Every item hash matches, but the recorded snapshot is different —
    # this can only happen if the caller deliberately altered state. The
    # plan must not claim no_changes.
    items_map = {it["item_id"]: item_content_hash(it) for it in items}
    doc = _state_doc(items=items_map, catalog_snapshot="sha256:oldoldoldoldold")
    plan = plan_incremental(items, state_read=_valid_state_read(doc))
    assert plan.items_to_encode == []
    assert plan.no_changes is False


# ============================================================================
# encode_lazily
# ============================================================================
class _FakeEncoder:
    """Deterministic synthetic encoder. Satisfies the Encoder protocol."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.model_version = "fake+v1"
        self.calls: list[list[str]] = []

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        rows = [_deterministic_unit_row(self._dim, t) for t in texts]
        return np.stack(rows).astype(np.float32)


def _deterministic_unit_row(dim: int, text: str) -> NDArray[np.float32]:
    """A deterministic, L2-normalized float32 vector seeded by ``text``."""
    seed = sum(text.encode("utf-8")) or 1
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return v.astype(np.float32)


def test_encode_lazily_yields_in_order() -> None:
    items: list[CatalogItem] = [
        {
            "item_id": f"i_{i:04d}",
            "title": f"T{i}",
            "description": "d",
            "category": "c",
            "brand": "b",
        }
        for i in range(5)
    ]
    enc = _FakeEncoder(dim=4)
    rows = list(encode_lazily(enc, items, batch_size=2, progress=False))
    assert len(rows) == 5
    # Batches were 2, 2, 1.
    assert [len(c) for c in enc.calls] == [2, 2, 1]


def test_encode_lazily_empty() -> None:
    enc = _FakeEncoder(dim=4)
    rows = list(encode_lazily(enc, [], batch_size=8, progress=False))
    assert rows == []


def test_encode_lazily_rejects_bad_batch_size() -> None:
    enc = _FakeEncoder(dim=4)
    with pytest.raises(ValueError, match="batch_size"):
        list(encode_lazily(enc, [], batch_size=0, progress=False))
