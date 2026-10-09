"""Unit tests for the in-memory popularity cache (ADR-0020 § Tier 4)."""

from __future__ import annotations

import json
import pathlib

import pytest

from recsys.popularity.cache import (
    SUPPORTED_SCHEMA_VERSION,
    PopularityCache,
    PopularityCacheError,
    SnapshotItem,
)


def _file(tmp_path: pathlib.Path, items: list[dict[str, object]]) -> pathlib.Path:
    p = tmp_path / "snapshot.json"
    p.write_text(
        json.dumps({"schema_version": SUPPORTED_SCHEMA_VERSION, "items": items}),
        encoding="utf-8",
    )
    return p


def _item(iid: str, rank: int, *, category: str = "books") -> dict[str, object]:
    return {
        "item_id": iid,
        "rank": rank,
        "category": category,
        "brand": "acme",
        "language": "en",
    }


def test_from_file_loads_items(tmp_path: pathlib.Path) -> None:
    p = _file(tmp_path, [_item("i_1", 1), _item("i_2", 2)])
    cache = PopularityCache.from_file(p)
    assert cache.size == 2


def test_from_file_missing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(PopularityCacheError, match="not found"):
        PopularityCache.from_file(tmp_path / "missing.json")


def test_from_file_invalid_json(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "snapshot.json"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(PopularityCacheError, match="not valid JSON"):
        PopularityCache.from_file(p)


def test_from_file_wrong_schema_version(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "snapshot.json"
    p.write_text(json.dumps({"schema_version": 999, "items": []}), encoding="utf-8")
    with pytest.raises(PopularityCacheError, match="schema_version"):
        PopularityCache.from_file(p)


def test_from_file_malformed_item(tmp_path: pathlib.Path) -> None:
    p = _file(tmp_path, [{"item_id": "i_1"}])
    with pytest.raises(PopularityCacheError, match="malformed"):
        PopularityCache.from_file(p)


def test_empty_returns_nothing() -> None:
    assert PopularityCache.empty().get(k=5) == []


def test_get_returns_items_ordered_by_rank() -> None:
    cache = PopularityCache(
        (
            SnapshotItem("i_1", 1, "books", "acme", "en"),
            SnapshotItem("i_2", 2, "books", "acme", "en"),
            SnapshotItem("i_3", 3, "books", "acme", "en"),
        )
    )
    out = cache.get(k=10)
    assert [iid for iid, _ in out] == ["i_1", "i_2", "i_3"]


def test_get_limits_to_k() -> None:
    cache = PopularityCache(
        tuple(SnapshotItem(f"i_{i}", i, "books", "acme", "en") for i in range(1, 11))
    )
    assert len(cache.get(k=3)) == 3


def test_scores_in_open_closed_unit_interval() -> None:
    cache = PopularityCache(
        tuple(SnapshotItem(f"i_{i}", i, "books", "acme", "en") for i in range(1, 6))
    )
    for _iid, score in cache.get(k=10):
        assert 0.0 < score <= 1.0


def test_score_decreases_with_rank() -> None:
    cache = PopularityCache(
        tuple(SnapshotItem(f"i_{i}", i, "books", "acme", "en") for i in range(1, 6))
    )
    scores = [s for _, s in cache.get(k=10)]
    assert scores == sorted(scores, reverse=True)


def test_filter_by_category() -> None:
    cache = PopularityCache(
        (
            SnapshotItem("i_1", 1, "books", "acme", "en"),
            SnapshotItem("i_2", 2, "music", "acme", "en"),
            SnapshotItem("i_3", 3, "books", "other", "en"),
        )
    )
    out = cache.get(k=10, filters={"category": "books"})
    assert [iid for iid, _ in out] == ["i_1", "i_3"]


def test_filter_multiple_fields() -> None:
    cache = PopularityCache(
        (
            SnapshotItem("i_1", 1, "books", "acme", "en"),
            SnapshotItem("i_2", 2, "books", "other", "en"),
        )
    )
    out = cache.get(k=10, filters={"category": "books", "brand": "acme"})
    assert [iid for iid, _ in out] == ["i_1"]


def test_filter_returns_empty_when_nothing_matches() -> None:
    cache = PopularityCache((SnapshotItem("i_1", 1, "books", "acme", "en"),))
    assert cache.get(k=5, filters={"category": "toys"}) == []


def test_unknown_filter_field_is_ignored() -> None:
    cache = PopularityCache((SnapshotItem("i_1", 1, "books", "acme", "en"),))
    out = cache.get(k=5, filters={"unknown": "x"})
    assert [iid for iid, _ in out] == ["i_1"]


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        PopularityCache.empty().get(k=0)
