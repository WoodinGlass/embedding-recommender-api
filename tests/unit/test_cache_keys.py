"""Unit tests for the cache key builder (ADR-0015)."""

from __future__ import annotations

import re

import pytest

from recsys.cache.keys import ALLOWED_ENDPOINTS, build_cache_key

_HEX32 = re.compile(r"^[0-9a-f]{32}$")


def _key(**overrides: object) -> str:
    base: dict[str, object] = {
        "index_version": "idx-deadbeef",
        "endpoint": "recommend",
        "k_max": 10,
        "filters": {"category": "books"},
        "seed_item_ids": ["i_1", "i_2"],
    }
    base.update(overrides)
    return build_cache_key(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- #
# shape
# ---------------------------------------------------------------- #
def test_key_has_namespace_version_endpoint_hash() -> None:
    # ADR-0015 amendment: prefix is recsys:cache so it cannot
    # collide with the rate limiter (recsys:rl:*).
    key = _key()
    parts = key.split(":")
    assert parts[0] == "recsys"
    assert parts[1] == "cache"
    assert parts[2] == "v1"
    assert parts[3] == "recommend"
    assert _HEX32.match(parts[4]), f"hash not 32 hex: {parts[4]!r}"
    assert len(parts) == 5


def test_variant_is_appended_not_hashed() -> None:
    key = _key(variant="control")
    parts = key.split(":")
    assert parts[-1] == "control"
    assert _HEX32.match(parts[4])


def test_no_variant_no_suffix() -> None:
    assert _key(variant=None).count(":") == 4


def test_both_endpoints_work() -> None:
    for endpoint in sorted(ALLOWED_ENDPOINTS):
        key = _key(endpoint=endpoint)
        assert f":{endpoint}:" in key


# ---------------------------------------------------------------- #
# determinism + sensitivity
# ---------------------------------------------------------------- #
def test_deterministic() -> None:
    assert _key() == _key()


def test_different_index_version_changes_key() -> None:
    assert _key(index_version="idx-1") != _key(index_version="idx-2")


def test_different_endpoint_changes_key() -> None:
    assert _key(endpoint="recommend") != _key(endpoint="similar")


def test_different_k_changes_key() -> None:
    assert _key(k_max=5) != _key(k_max=10)


def test_different_filters_change_key() -> None:
    assert _key(filters={"category": "books"}) != _key(filters={"category": "music"})


def test_different_seeds_change_key() -> None:
    assert _key(seed_item_ids=["i_1"]) != _key(seed_item_ids=["i_2"])


# ---------------------------------------------------------------- #
# canonicalization
# ---------------------------------------------------------------- #
def test_filter_key_order_is_irrelevant() -> None:
    a = _key(filters={"category": "books", "brand": "acme"})
    b = _key(filters={"brand": "acme", "category": "books"})
    assert a == b


def test_seed_order_is_irrelevant() -> None:
    assert _key(seed_item_ids=["i_1", "i_2"]) == _key(seed_item_ids=["i_2", "i_1"])


def test_none_filters_equals_empty_filters() -> None:
    assert _key(filters=None) == _key(filters={})


# ---------------------------------------------------------------- #
# guards
# ---------------------------------------------------------------- #
def test_unknown_endpoint_raises() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        _key(endpoint="admin")


def test_zero_k_raises() -> None:
    with pytest.raises(ValueError, match="k"):
        _key(k_max=0)


def test_empty_index_version_raises() -> None:
    with pytest.raises(ValueError, match="index_version"):
        _key(index_version="")


# ---------------------------------------------------------------- #
# negative-space
# ---------------------------------------------------------------- #
def test_key_does_not_leak_inputs() -> None:
    """The hash is over the payload; no raw value appears in the key."""
    key = _key(filters={"category": "very-distinctive-value"})
    assert "very-distinctive-value" not in key
    assert "idx-deadbeef" not in key
