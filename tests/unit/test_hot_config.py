"""Unit tests for the hot config loader and store (ADR-0022)."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from recsys.config.hot import (
    HotConfigError,
    HotConfigStore,
    RateLimitSection,
    load_hot_config,
)

VALID: dict[str, Any] = {
    "schema_version": 1,
    "rate_limit": {
        "recommend_per_minute": 600,
        "events_per_minute": 6000,
        "admin_per_minute": 60,
    },
    "rerank": {
        "w_sim": 0.7,
        "w_pop": 0.2,
        "w_rec": 0.1,
        "mmr_lambda": 0.7,
        "mmr_window": 50,
        "mmr_min_k": 10,
        "candidate_multiplier": 4,
        "recency_half_life_days": 90,
    },
    "cache": {
        "recommend_ttl_seconds": 300,
        "similar_ttl_seconds": 600,
        "negative_ttl_seconds": 30,
    },
}


def _variant(**overrides: Any) -> dict[str, Any]:
    out = copy.deepcopy(VALID)
    out.update(overrides)
    return out


def _variant_nested(section: str, **overrides: Any) -> dict[str, Any]:
    out = copy.deepcopy(VALID)
    out[section].update(overrides)
    return out


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    p = tmp_path / "hot.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _bump_mtime(p: Path) -> None:
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


# ---------------------------------------------------------------- #
# load_hot_config
# ---------------------------------------------------------------- #
def test_load_valid(tmp_path: Path) -> None:
    cfg = load_hot_config(_write(tmp_path, VALID))
    assert cfg.schema_version == 1
    assert cfg.rate_limit.recommend_per_minute == 600
    assert cfg.rate_limit.events_per_minute == 6000
    assert cfg.rate_limit.admin_per_minute == 60
    assert cfg.rerank.w_sim == 0.7
    assert cfg.rerank.candidate_multiplier == 4
    assert cfg.cache.recommend_ttl_seconds == 300


def test_rate_limit_as_class_map() -> None:
    section = RateLimitSection(
        recommend_per_minute=600,
        events_per_minute=6000,
        admin_per_minute=60,
    )
    assert section.as_class_map() == {
        "recommend": 600,
        "events": 6000,
        "admin": 60,
    }


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(HotConfigError, match="not found"):
        load_hot_config(tmp_path / "missing.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "hot.yaml"
    p.write_text("not: [valid: yaml", encoding="utf-8")
    with pytest.raises(HotConfigError, match="not valid YAML"):
        load_hot_config(p)


def test_root_not_mapping_raises(tmp_path: Path) -> None:
    p = tmp_path / "hot.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(HotConfigError, match="root must be a mapping"):
        load_hot_config(p)


def test_wrong_schema_version_raises(tmp_path: Path) -> None:
    with pytest.raises(HotConfigError, match="schema_version"):
        load_hot_config(_write(tmp_path, _variant(schema_version=999)))


@pytest.mark.parametrize("key", ["secret", "password", "token", "api_key", "salt"])
def test_secret_marker_rejected(tmp_path: Path, key: str) -> None:
    with pytest.raises(HotConfigError, match="looks like a secret"):
        load_hot_config(_write(tmp_path, _variant(extra={key: "value"})))


def test_secret_marker_detected_in_nested_section(tmp_path: Path) -> None:
    data = _variant_nested("rate_limit", api_key="x")
    with pytest.raises(HotConfigError, match="looks like a secret"):
        load_hot_config(_write(tmp_path, data))


def test_missing_required_field_raises(tmp_path: Path) -> None:
    data = _variant(rate_limit={"recommend_per_minute": 600})
    with pytest.raises(HotConfigError, match="events_per_minute"):
        load_hot_config(_write(tmp_path, data))


def test_zero_limit_rejected(tmp_path: Path) -> None:
    data = _variant_nested("rate_limit", recommend_per_minute=0)
    with pytest.raises(HotConfigError, match="recommend_per_minute"):
        load_hot_config(_write(tmp_path, data))


def test_bool_is_not_int(tmp_path: Path) -> None:
    data = _variant_nested("cache", recommend_ttl_seconds=True)
    with pytest.raises(HotConfigError, match="must be an int"):
        load_hot_config(_write(tmp_path, data))


# ---------------------------------------------------------------- #
# HotConfigStore
# ---------------------------------------------------------------- #
def test_store_from_path(tmp_path: Path) -> None:
    store = HotConfigStore.from_path(_write(tmp_path, VALID))
    assert store.get().rate_limit.recommend_per_minute == 600


def test_store_from_path_propagates_first_load_error(tmp_path: Path) -> None:
    with pytest.raises(HotConfigError):
        HotConfigStore.from_path(tmp_path / "missing.yaml")


def test_store_reload_if_changed_no_change_returns_false(tmp_path: Path) -> None:
    store = HotConfigStore.from_path(_write(tmp_path, VALID))
    assert store.reload_if_changed() is False


def test_store_reload_if_changed_swaps_on_mtime_change(tmp_path: Path) -> None:
    p = _write(tmp_path, VALID)
    store = HotConfigStore.from_path(p)
    time.sleep(0.01)
    p.write_text(
        yaml.safe_dump(_variant_nested("rate_limit", recommend_per_minute=1200)),
        encoding="utf-8",
    )
    _bump_mtime(p)
    assert store.reload_if_changed() is True
    assert store.get().rate_limit.recommend_per_minute == 1200


def test_store_reload_keeps_previous_on_malformed(tmp_path: Path) -> None:
    p = _write(tmp_path, VALID)
    store = HotConfigStore.from_path(p)
    before = store.get()
    time.sleep(0.01)
    p.write_text("not: [valid: yaml", encoding="utf-8")
    _bump_mtime(p)
    assert store.reload_if_changed() is False
    assert store.get() is before


def test_store_reload_ignores_second_failure_same_mtime(tmp_path: Path) -> None:
    p = _write(tmp_path, VALID)
    store = HotConfigStore.from_path(p)
    p.write_text("not: [valid", encoding="utf-8")
    _bump_mtime(p)
    assert store.reload_if_changed() is False
    assert store.reload_if_changed() is False
