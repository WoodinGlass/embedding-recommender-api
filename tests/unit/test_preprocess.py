"""Unit tests for preprocessing rules.

These tests are the fastest tier: no model, no onnxruntime, no network.
They lock the *rules*, not just the code: if someone changes a rule without
bumping PREPROCESSING_VERSION, the hash tests will fail and force the bump.
"""

from __future__ import annotations

import hashlib

import pytest

from recsys.embeddings.preprocess import (
    MAX_CHARS,
    PREPROCESSING_VERSION,
    CatalogItem,
    catalog_snapshot,
    item_content_hash,
    preprocess_item,
)


def _item(**over: str) -> CatalogItem:
    base: CatalogItem = {
        "item_id": "i_0001",
        "title": "Dune",
        "description": "A desert planet epic.",
        "category": "books",
        "brand": "ace_books",
    }
    base.update(over)  # type: ignore[typeddict-item]
    return base


def test_version_is_pinned() -> None:
    assert PREPROCESSING_VERSION == "v1"


def test_title_and_description_joined_with_pipe() -> None:
    out = preprocess_item(_item(title="A", description="B"))
    assert out == "A | B"


def test_empty_description_omits_separator() -> None:
    assert preprocess_item(_item(title="A", description="")) == "A"
    assert preprocess_item(_item(title="A", description="   ")) == "A"


def test_whitespace_is_collapsed() -> None:
    out = preprocess_item(_item(title="  A   B  ", description="C\n\nD\tE"))
    assert out == "A B | C D E"


def test_unicode_is_nfkc_normalized() -> None:
    # U+FB01 LATIN SMALL LIGATURE FI normalizes under NFKC to "fi".
    out = preprocess_item(_item(title="\ufb01le", description=""))
    assert out == "file"


def test_truncation_respects_max_chars() -> None:
    long = "x" * (MAX_CHARS * 2)
    assert len(preprocess_item(_item(title=long, description=""))) == MAX_CHARS


def test_strip_is_applied_after_truncation() -> None:
    out = preprocess_item(_item(title="   hi   ", description="   "))
    assert out == "hi"


def test_content_hash_is_16_hex_and_deterministic() -> None:
    h1 = item_content_hash(_item())
    h2 = item_content_hash(_item())
    assert h1 == h2
    assert len(h1) == 16
    assert all(c in "0123456789abcdef" for c in h1)


def test_content_hash_changes_with_any_field() -> None:
    base = item_content_hash(_item())
    # item_id and brand do not appear in the preprocessed text, so they
    # legitimately do not change the content hash. Title, description, and
    # category do.
    assert item_content_hash(_item(title="Other")) != base
    assert item_content_hash(_item(description="Other")) != base


def test_content_hash_matches_the_documented_formula() -> None:
    item = _item(title="A", description="B")
    expected_input = b"A | B"
    expected = hashlib.sha256(expected_input).hexdigest()[:16]
    assert item_content_hash(item) == expected


def test_catalog_snapshot_format() -> None:
    snap = catalog_snapshot([_item()])
    assert snap.startswith("sha256:")
    assert len(snap) == len("sha256:") + 16
    assert all(c in "0123456789abcdef" for c in snap[len("sha256:") :])


def test_catalog_snapshot_ignores_row_order() -> None:
    a = [_item(item_id="i_0002", title="B"), _item(item_id="i_0001", title="A")]
    b = [_item(item_id="i_0001", title="A"), _item(item_id="i_0002", title="B")]
    assert catalog_snapshot(a) == catalog_snapshot(b)


def test_catalog_snapshot_changes_when_content_changes() -> None:
    a = catalog_snapshot([_item(title="A"), _item(item_id="i_0002", title="B")])
    b = catalog_snapshot([_item(title="A"), _item(item_id="i_0002", title="B!")])
    assert a != b


def test_catalog_snapshot_accepts_duplicate_ids_by_full_pair_sort() -> None:
    # If two items share an id but differ in content, sorting by the full
    # (item_id, content_hash) tuple keeps the result deterministic.
    items = [_item(item_id="dup", title="A"), _item(item_id="dup", title="B")]
    s1 = catalog_snapshot(items)
    s2 = catalog_snapshot(list(reversed(items)))
    assert s1 == s2


@pytest.mark.parametrize(
    ("title", "description"),
    [
        ("", ""),
        (" ", " "),
        ("a", ""),
        ("\u00e9", "caf\u00e9"),  # accented characters survive NFKC unchanged
        ("emoji \U0001f600", "ok"),
    ],
)
def test_never_raises_on_edges(title: str, description: str) -> None:
    out = preprocess_item(_item(title=title, description=description))
    assert isinstance(out, str)
    assert not out.startswith(" | ")
    assert not out.endswith(" | ")
