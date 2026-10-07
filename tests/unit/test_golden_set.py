"""Unit tests for the golden set loader.

Pure module: reads a file, validates it, returns a dataclass. No
database, no configuration. These tests pin the validation rules from
ADR-0009 § 1, including the ones that exist to catch data-entry mistakes
rather than crashes.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from recsys.evaluation.golden_set import (
    DEFAULT_GOLDEN_SET_VERSION,
    GoldenQuery,
    GoldenSetError,
    default_golden_set_path,
    load_golden_set,
    version_from_filename,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _write_jsonl(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _row(
    query_id: str = "q1",
    topic: str = "t1",
    seeds: list[str] | None = None,
    relevant: list[str] | None = None,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "topic": topic,
        "seed_item_ids": seeds if seeds is not None else ["i_0001"],
        "relevant_item_ids": relevant if relevant is not None else ["i_0002"],
    }


@pytest.fixture
def v1_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "v1.jsonl"


# --------------------------------------------------------------------------- #
# version_from_filename
# --------------------------------------------------------------------------- #
def test_version_from_filename_v1() -> None:
    assert version_from_filename(pathlib.Path("v1.jsonl")) == "v1"


def test_version_from_filename_v42() -> None:
    assert version_from_filename(pathlib.Path("/a/b/v42.jsonl")) == "v42"


@pytest.mark.parametrize(
    "bad",
    ["golden.jsonl", "v1.yaml", "v1.JSONL", "queries.jsonl", "v1.jsonl.bak"],
)
def test_version_from_filename_rejects(bad: str) -> None:
    with pytest.raises(GoldenSetError, match="must match"):
        version_from_filename(pathlib.Path(bad))


# --------------------------------------------------------------------------- #
# load_golden_set — happy path
# --------------------------------------------------------------------------- #
def test_load_happy_path(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(), _row(query_id="q2")])
    gs = load_golden_set(v1_path)
    assert gs.version == "v1"
    assert len(gs) == 2
    assert gs.queries[0].query_id == "q1"
    assert gs.queries[0].seed_item_ids == ("i_0001",)
    assert gs.queries[0].relevant_item_ids == ("i_0002",)


def test_load_skips_blank_lines(v1_path: pathlib.Path) -> None:
    v1_path.write_text("\n" + json.dumps(_row()) + "\n\n", encoding="utf-8")
    gs = load_golden_set(v1_path)
    assert len(gs) == 1


def test_load_accepts_known_item_ids(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(seeds=["i_0001"], relevant=["i_0002"])])
    gs = load_golden_set(v1_path, known_item_ids=["i_0001", "i_0002"])
    assert len(gs) == 1


def test_query_is_frozen(v1_path: pathlib.Path) -> None:
    from dataclasses import FrozenInstanceError

    _write_jsonl(v1_path, [_row()])
    gs = load_golden_set(v1_path)
    with pytest.raises(FrozenInstanceError):
        gs.queries[0].topic = "x"  # type: ignore[misc]


def test_golden_query_is_a_dataclass() -> None:
    q = GoldenQuery(
        query_id="q",
        topic="t",
        seed_item_ids=("a",),
        relevant_item_ids=("b",),
    )
    assert q.query_id == "q"


# --------------------------------------------------------------------------- #
# load_golden_set — file errors
# --------------------------------------------------------------------------- #
def test_load_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(GoldenSetError, match="not found"):
        load_golden_set(tmp_path / "v1.jsonl")


def test_load_empty_file(v1_path: pathlib.Path) -> None:
    v1_path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="no queries"):
        load_golden_set(v1_path)


def test_load_malformed_json(v1_path: pathlib.Path) -> None:
    v1_path.write_text("{ not json\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="invalid JSON"):
        load_golden_set(v1_path)


def test_load_not_json_object(v1_path: pathlib.Path) -> None:
    v1_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="not a JSON object"):
        load_golden_set(v1_path)


# --------------------------------------------------------------------------- #
# load_golden_set — field validation
# --------------------------------------------------------------------------- #
def test_missing_field(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [{"query_id": "q1", "topic": "t1"}])
    with pytest.raises(GoldenSetError, match="missing fields"):
        load_golden_set(v1_path)


def test_extra_field(v1_path: pathlib.Path) -> None:
    row = {**_row(), "extra": "x"}
    _write_jsonl(v1_path, [row])
    with pytest.raises(GoldenSetError, match="unexpected fields"):
        load_golden_set(v1_path)


def test_empty_query_id(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(query_id="")])
    with pytest.raises(GoldenSetError, match="query_id must be a non-empty"):
        load_golden_set(v1_path)


def test_duplicate_query_id(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(query_id="q"), _row(query_id="q")])
    with pytest.raises(GoldenSetError, match="duplicate query_id"):
        load_golden_set(v1_path)


def test_empty_topic(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(topic="")])
    with pytest.raises(GoldenSetError, match="topic must be a non-empty"):
        load_golden_set(v1_path)


def test_seed_item_ids_not_a_list(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [{**_row(), "seed_item_ids": "not-a-list"}])
    with pytest.raises(GoldenSetError, match="seed_item_ids must be a list"):
        load_golden_set(v1_path)


def test_seed_item_ids_empty(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(seeds=[])])
    with pytest.raises(GoldenSetError, match="at least one id"):
        load_golden_set(v1_path)


def test_seed_item_ids_non_string(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [{**_row(), "seed_item_ids": ["i_0001", 42]}])
    with pytest.raises(GoldenSetError, match=r"seed_item_ids\[1\]"):
        load_golden_set(v1_path)


def test_seed_item_ids_empty_string(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [{**_row(), "seed_item_ids": ["i_0001", ""]}])
    with pytest.raises(GoldenSetError, match=r"seed_item_ids\[1\]"):
        load_golden_set(v1_path)


def test_seed_item_ids_duplicate(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(seeds=["i_0001", "i_0001"])])
    with pytest.raises(GoldenSetError, match="duplicate"):
        load_golden_set(v1_path)


def test_relevant_item_ids_not_a_list(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [{**_row(), "relevant_item_ids": "x"}])
    with pytest.raises(GoldenSetError, match="relevant_item_ids must be a list"):
        load_golden_set(v1_path)


def test_seed_and_relevant_overlap(v1_path: pathlib.Path) -> None:
    _write_jsonl(
        v1_path,
        [_row(seeds=["i_0001", "i_0002"], relevant=["i_0002"])],
    )
    with pytest.raises(GoldenSetError, match="overlap"):
        load_golden_set(v1_path)


# --------------------------------------------------------------------------- #
# load_golden_set — known_item_ids
# --------------------------------------------------------------------------- #
def test_unknown_seed_item_id(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(seeds=["i_9999"])])
    with pytest.raises(GoldenSetError, match="unknown item id"):
        load_golden_set(v1_path, known_item_ids=["i_0001", "i_0002"])


def test_unknown_relevant_item_id(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(relevant=["i_9999"])])
    with pytest.raises(GoldenSetError, match="unknown item id"):
        load_golden_set(v1_path, known_item_ids=["i_0001"])


def test_known_item_ids_none_skips_cross_check(v1_path: pathlib.Path) -> None:
    _write_jsonl(v1_path, [_row(seeds=["i_9999"], relevant=["i_8888"])])
    # No exception when known_item_ids is None.
    gs = load_golden_set(v1_path)
    assert gs.queries[0].seed_item_ids == ("i_9999",)


# --------------------------------------------------------------------------- #
# default_golden_set_path
# --------------------------------------------------------------------------- #
def test_default_golden_set_path() -> None:
    p = default_golden_set_path(pathlib.Path("/repo"))
    assert p == pathlib.Path("/repo/evaluation/golden_set/v1.jsonl")


def test_default_golden_set_path_version() -> None:
    p = default_golden_set_path(pathlib.Path("/repo"), version="v2")
    assert p == pathlib.Path("/repo/evaluation/golden_set/v2.jsonl")


def test_default_version_constant() -> None:
    assert DEFAULT_GOLDEN_SET_VERSION == "v1"
