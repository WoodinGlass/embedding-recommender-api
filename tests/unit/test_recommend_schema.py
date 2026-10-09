"""Contract tests for the RecommendResponse schema (ADR-0020).

The five-value `meta.source` enum is locked here. A change to it is a
change to the API contract; the test fails, and the fix is to update
ADR-0020 and `docs/contracts.md` § 2.1 together with the enum, not to
loosen the test.
"""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from recsys.api.schemas.recommend import (
    RecommendItem,
    RecommendMeta,
    RecommendResponse,
)

EXPECTED_SOURCES = {
    "cache",
    "ann",
    "fallback_ann",
    "fallback_cached",
    "none",
}


def _meta(**overrides: object) -> RecommendMeta:
    base: dict[str, object] = {
        "source": "ann",
        "model_version": "m1",
        "index_version": "idx-1",
    }
    base.update(overrides)
    return RecommendMeta(**base)  # type: ignore[arg-type]


def test_source_enum_is_exactly_five_values() -> None:
    hints = typing.get_type_hints(RecommendMeta)
    source_type = hints["source"]
    actual = set(typing.get_args(source_type))
    assert actual == EXPECTED_SOURCES, (
        f"meta.source values changed: {actual!r}. "
        "Update ADR-0020 and docs/contracts.md § 2.1 in the same change."
    )


@pytest.mark.parametrize("value", sorted(EXPECTED_SOURCES))
def test_each_source_value_accepted(value: str) -> None:
    m = _meta(source=value)
    assert m.source == value


@pytest.mark.parametrize("value", ["fallback", "popularity", "cached", "", "ANN"])
def test_unknown_source_value_rejected(value: str) -> None:
    with pytest.raises(ValidationError):
        _meta(source=value)


def test_experiment_defaults_to_none() -> None:
    assert _meta().experiment is None


def test_experiment_null_for_similar_shape() -> None:
    m = _meta(experiment=None)
    assert m.experiment is None


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        _meta(unknown="x")


def test_response_has_request_id_items_meta() -> None:
    r = RecommendResponse(
        request_id="abc",
        items=[RecommendItem(item_id="i_1", score=0.9, rank=1)],
        meta=_meta(),
    )
    assert r.request_id == "abc"
    assert r.items[0].item_id == "i_1"
    assert r.meta.source == "ann"


def test_item_score_is_bounded() -> None:
    with pytest.raises(ValidationError):
        RecommendItem(item_id="i_1", score=1.5, rank=1)
    with pytest.raises(ValidationError):
        RecommendItem(item_id="i_1", score=-0.1, rank=1)


def test_item_rank_starts_at_one() -> None:
    with pytest.raises(ValidationError):
        RecommendItem(item_id="i_1", score=0.5, rank=0)
