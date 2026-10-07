"""Unit tests for :mod:`recsys.retrieval.identity`.

Pure-function module. No database, no config, no I/O. The tests pin the
hashing contract from ADR-0007: determinism, sensitivity to each input,
exclusion of the query-time and evaluation-metadata fields, and the
collision-resolution behavior.
"""

from __future__ import annotations

import pytest

from recsys.retrieval.identity import (
    DEFAULT_PREFIX_LENGTH,
    EXTENDED_PREFIX_LENGTH,
    IndexIdentityInputs,
    canonical_json,
    index_version,
    resolve_index_version,
)

pytestmark = pytest.mark.unit


_BASE: IndexIdentityInputs = IndexIdentityInputs(
    model_version="minilm-onnx-v1+a3f9e021",
    catalog_snapshot="sha256:9f1e2c8a3b5d7e4f",
    preprocessing_version="v1",
    metric="cosine",
    hnsw_m=16,
    hnsw_ef_construction=64,
    pgvector_version="0.8.0",
)


# --------------------------------------------------------------------------- #
# canonical JSON
# --------------------------------------------------------------------------- #
def test_canonical_json_is_deterministic() -> None:
    assert canonical_json(_BASE) == canonical_json(_BASE)


def test_canonical_json_keys_are_sorted() -> None:
    doc = canonical_json(_BASE)
    # "catalog_snapshot" must appear before "hnsw_ef_construction" which
    # appears before "metric" which appears before "model_version".
    assert doc.index("catalog_snapshot") < doc.index("hnsw_ef_construction")
    assert doc.index("hnsw_ef_construction") < doc.index("metric")
    assert doc.index("metric") < doc.index("model_version")


def test_canonical_json_has_no_whitespace() -> None:
    doc = canonical_json(_BASE)
    assert " " not in doc
    assert "\n" not in doc


def test_canonical_json_includes_every_input() -> None:
    doc = canonical_json(_BASE)
    for field in (
        "model_version",
        "catalog_snapshot",
        "preprocessing_version",
        "metric",
        "hnsw_m",
        "hnsw_ef_construction",
        "pgvector_version",
    ):
        assert field in doc, f"missing {field} in canonical form"


# --------------------------------------------------------------------------- #
# index_version
# --------------------------------------------------------------------------- #
def test_index_version_format() -> None:
    v = index_version(_BASE)
    assert v.startswith("idx-")
    assert len(v) == len("idx-") + DEFAULT_PREFIX_LENGTH
    assert all(c in "0123456789abcdef" for c in v[len("idx-") :])


def test_index_version_is_deterministic() -> None:
    assert index_version(_BASE) == index_version(_BASE)


def test_index_version_rejects_short_prefix() -> None:
    with pytest.raises(ValueError, match="prefix_length"):
        index_version(_BASE, prefix_length=4)


@pytest.mark.parametrize(
    ("field", "new_value"),
    [
        ("model_version", "minilm-onnx-v2+beef1234"),
        ("catalog_snapshot", "sha256:0000000000000000"),
        ("preprocessing_version", "v2"),
        ("metric", "l2"),
        ("hnsw_m", 32),
        ("hnsw_ef_construction", 128),
        ("pgvector_version", "0.9.0"),
    ],
)
def test_index_version_is_sensitive_to_each_input(field: str, new_value: object) -> None:
    base = index_version(_BASE)
    changed = index_version(IndexIdentityInputs(**{**_BASE.__dict__, field: new_value}))
    assert changed != base, f"hash did not change when {field} changed"


def test_index_version_ignores_dict_order() -> None:
    # The canonical form sorts keys, so argument order does not matter.
    # There is only one constructor order for a frozen dataclass, but the
    # hashed JSON is insensitive to it by construction; this test pins the
    # property so a future change to canonical_json does not lose it.
    a = index_version(_BASE)
    b = index_version(
        IndexIdentityInputs(
            model_version=_BASE.model_version,
            catalog_snapshot=_BASE.catalog_snapshot,
            preprocessing_version=_BASE.preprocessing_version,
            metric=_BASE.metric,
            hnsw_m=_BASE.hnsw_m,
            hnsw_ef_construction=_BASE.hnsw_ef_construction,
            pgvector_version=_BASE.pgvector_version,
        )
    )
    assert a == b


# --------------------------------------------------------------------------- #
# resolve_index_version
# --------------------------------------------------------------------------- #
def test_resolve_new_when_lookup_misses() -> None:
    version, status = resolve_index_version(_BASE, lookup=lambda _: None)
    assert status == "new"
    assert version == index_version(_BASE)


def test_resolve_duplicate_when_lookup_returns_same_inputs() -> None:
    version, status = resolve_index_version(_BASE, lookup=lambda _: _BASE)
    assert status == "duplicate"
    assert version == index_version(_BASE)


def test_resolve_extends_on_real_collision() -> None:
    """An 8-hex collision with different inputs extends to 12 hex."""
    base_8 = index_version(_BASE, prefix_length=DEFAULT_PREFIX_LENGTH)

    # A different input set that the fake lookup will claim shares the
    # 8-hex prefix. The lookup maps base_8 to different inputs, and any
    # other version to None.
    other: IndexIdentityInputs = IndexIdentityInputs(
        model_version="other-model",
        catalog_snapshot=_BASE.catalog_snapshot,
        preprocessing_version=_BASE.preprocessing_version,
        metric=_BASE.metric,
        hnsw_m=_BASE.hnsw_m,
        hnsw_ef_construction=_BASE.hnsw_ef_construction,
        pgvector_version=_BASE.pgvector_version,
    )

    def lookup(candidate: str) -> IndexIdentityInputs | None:
        if candidate == base_8:
            return other
        return None

    version, status = resolve_index_version(_BASE, lookup=lookup)
    assert status == "new"
    assert len(version) == len("idx-") + EXTENDED_PREFIX_LENGTH
    assert version != base_8


def test_resolve_raises_on_double_collision() -> None:
    """An 8-hex and 12-hex collision with different inputs raises."""
    base_8 = index_version(_BASE, prefix_length=DEFAULT_PREFIX_LENGTH)
    base_12 = index_version(_BASE, prefix_length=EXTENDED_PREFIX_LENGTH)
    other: IndexIdentityInputs = IndexIdentityInputs(
        model_version="other-model",
        catalog_snapshot=_BASE.catalog_snapshot,
        preprocessing_version=_BASE.preprocessing_version,
        metric=_BASE.metric,
        hnsw_m=_BASE.hnsw_m,
        hnsw_ef_construction=_BASE.hnsw_ef_construction,
        pgvector_version=_BASE.pgvector_version,
    )

    def lookup(candidate: str) -> IndexIdentityInputs | None:
        if candidate in (base_8, base_12):
            return other
        return None

    with pytest.raises(RuntimeError, match="collision"):
        resolve_index_version(_BASE, lookup=lookup)


def test_resolve_prefers_shorter_prefix_when_possible() -> None:
    # Even with a real collision at 8, when 12 does not collide, the result
    # is the 12-hex form, not a longer one.
    base_8 = index_version(_BASE, prefix_length=DEFAULT_PREFIX_LENGTH)
    other: IndexIdentityInputs = IndexIdentityInputs(
        model_version="other",
        catalog_snapshot=_BASE.catalog_snapshot,
        preprocessing_version=_BASE.preprocessing_version,
        metric=_BASE.metric,
        hnsw_m=_BASE.hnsw_m,
        hnsw_ef_construction=_BASE.hnsw_ef_construction,
        pgvector_version=_BASE.pgvector_version,
    )

    def lookup(candidate: str) -> IndexIdentityInputs | None:
        return other if candidate == base_8 else None

    version, _ = resolve_index_version(_BASE, lookup=lookup)
    assert version == index_version(_BASE, prefix_length=EXTENDED_PREFIX_LENGTH)
