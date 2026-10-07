"""Unit tests for the filter-field allowlist and its validator.

``validate_filters`` is a pure function: no I/O, no configuration. The
tests here pin its contract — what counts as "no filter", how unknown
fields are reported, and why the returned mapping has a stable key order.
"""

from __future__ import annotations

import pytest

from recsys.retrieval.filters import (
    FILTER_FIELDS,
    UnknownFilterFieldError,
    validate_filters,
)

pytestmark = pytest.mark.unit


def test_filter_fields_is_the_documented_set() -> None:
    # The set is fixed by docs/contracts.md § 2.1. Changing it is a
    # contract change and must update that section in the same PR.
    assert FILTER_FIELDS == ("brand", "category", "language")


@pytest.mark.parametrize("value", [None, {}])
def test_empty_input_returns_none(value: object) -> None:
    assert validate_filters(value) is None  # type: ignore[arg-type]


def test_single_field_passes_through() -> None:
    result = validate_filters({"category": "books"})
    assert result == {"category": "books"}


def test_multiple_fields_pass_through() -> None:
    result = validate_filters({"category": "books", "brand": "ace", "language": "en"})
    assert result == {"category": "books", "brand": "ace", "language": "en"}


def test_unknown_field_raises_with_helpful_message() -> None:
    with pytest.raises(UnknownFilterFieldError, match="unknown filter field"):
        validate_filters({"color": "red"})


def test_unknown_field_lists_the_allowlist() -> None:
    with pytest.raises(UnknownFilterFieldError) as excinfo:
        validate_filters({"color": "red"})
    message = str(excinfo.value)
    for field in FILTER_FIELDS:
        assert field in message


def test_multiple_unknown_fields_all_named() -> None:
    with pytest.raises(UnknownFilterFieldError) as excinfo:
        validate_filters({"color": "red", "size": "L"})
    message = str(excinfo.value)
    assert "color" in message
    assert "size" in message


def test_non_string_value_raises() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        validate_filters({"category": 42})  # type: ignore[dict-item]


def test_empty_string_values_are_dropped() -> None:
    assert validate_filters({"category": ""}) is None
    assert validate_filters({"category": "   "}) is None


def test_mixed_present_and_empty_keeps_present() -> None:
    result = validate_filters({"category": "books", "brand": ""})
    assert result == {"category": "books"}


def test_whitespace_is_stripped() -> None:
    result = validate_filters({"category": "  books  "})
    assert result == {"category": "books"}


def test_result_key_order_follows_filter_fields() -> None:
    # Deterministic key order regardless of input order — matters for
    # hashing the filter set into a cache key (M3).
    a = validate_filters({"language": "en", "category": "books", "brand": "ace"})
    b = validate_filters({"brand": "ace", "category": "books", "language": "en"})
    assert a == b
    assert list(a.keys()) == ["brand", "category", "language"]  # type: ignore[union-attr]


def test_returns_a_new_dict_not_the_input() -> None:
    # Mutating the result must not affect the caller's mapping.
    original = {"category": "books"}
    result = validate_filters(original)
    assert result is not None
    result["category"] = "music"
    assert original == {"category": "books"}


def test_pure_function_no_state_between_calls() -> None:
    # Two identical calls return equal (but not necessarily identical)
    # results; no hidden state.
    assert validate_filters({"category": "books"}) == validate_filters({"category": "books"})
