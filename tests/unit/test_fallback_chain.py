"""Unit tests for the tier-3 / tier-4 fallback chain (ADR-0020).

Both tiers are faked: ``read_snapshot`` is monkeypatched on the
``recsys.fallback.chain`` module so the chain's logic (order, empty
handling, exception handling, ``FallbackResult`` shape) is what is
under test, not the SQL or the JSON loader — those have their own
tests (``test_popularity_snapshot_reader``, and the cache's own unit
tests if any).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from recsys.fallback import chain as chain_module
from recsys.fallback.chain import FallbackResult, serve_fallback
from recsys.popularity import PopularityCache, SnapshotItem

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- #
# fakes
# ---------------------------------------------------------------- #
class _FakeConnection:
    """Opaque. ``serve_fallback`` only passes it to ``read_snapshot``,
    which is monkeypatched in these tests."""


def _cache(items: list[SnapshotItem]) -> PopularityCache:
    return PopularityCache(tuple(items))


def _snapshot_item(item_id: str, rank: int) -> SnapshotItem:
    return SnapshotItem(
        item_id=item_id,
        rank=rank,
        category="books",
        brand="acme",
        language="en",
    )


def _stub_read_snapshot(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[tuple[Any, Any]]:
    """Replace ``read_snapshot`` on the chain module. Return the list
    that records each call so a test can assert on arguments."""
    calls: list[tuple[Any, Any]] = []

    def _stub(
        connection: Any,
        *,
        k: int,
        filters: Mapping[str, str] | None = None,
        timeout_seconds: float,
    ) -> Any:
        calls.append((connection, (k, dict(filters or {}), timeout_seconds)))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(chain_module, "read_snapshot", _stub)
    return calls


# ---------------------------------------------------------------- #
# tier 3 wins
# ---------------------------------------------------------------- #
def test_tier3_nonempty_returns_fallback_ann(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_read_snapshot(monkeypatch, [("i_1", 0.9), ("i_2", 0.5)])
    result = serve_fallback(
        connection=_FakeConnection(),
        popularity_cache=_cache([_snapshot_item("i_cache", 1)]),
        k=2,
        filters=None,
    )
    assert isinstance(result, FallbackResult)
    assert result.source == "fallback_ann"
    assert result.items == (("i_1", 0.9), ("i_2", 0.5))


def test_tier3_receives_k_filters_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConnection()
    calls = _stub_read_snapshot(monkeypatch, [("i_1", 1.0)])
    serve_fallback(
        connection=conn,
        popularity_cache=_cache([]),
        k=7,
        filters={"category": "books"},
        timeout_seconds=0.5,
    )
    assert calls
    got_conn, (k, filters, timeout) = calls[0]
    assert got_conn is conn
    assert k == 7
    assert filters == {"category": "books"}
    assert timeout == 0.5


# ---------------------------------------------------------------- #
# tier 3 empty -> tier 4
# ---------------------------------------------------------------- #
def test_tier3_empty_falls_through_to_tier4(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_read_snapshot(monkeypatch, [])
    result = serve_fallback(
        connection=_FakeConnection(),
        popularity_cache=_cache([_snapshot_item("i_cache", 1)]),
        k=2,
        filters=None,
    )
    assert isinstance(result, FallbackResult)
    assert result.source == "fallback_cached"
    assert result.items[0][0] == "i_cache"


# ---------------------------------------------------------------- #
# tier 3 exception -> tier 4
# ---------------------------------------------------------------- #
def test_tier3_raises_falls_through_to_tier4(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_read_snapshot(monkeypatch, RuntimeError("db down"))
    result = serve_fallback(
        connection=_FakeConnection(),
        popularity_cache=_cache([_snapshot_item("i_cache", 1)]),
        k=2,
        filters=None,
    )
    assert isinstance(result, FallbackResult)
    assert result.source == "fallback_cached"


# ---------------------------------------------------------------- #
# connection None -> skip tier 3
# ---------------------------------------------------------------- #
def test_connection_none_skips_tier3(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_read_snapshot(monkeypatch, [("i_1", 1.0)])
    result = serve_fallback(
        connection=None,
        popularity_cache=_cache([_snapshot_item("i_cache", 1)]),
        k=2,
        filters=None,
    )
    assert not calls, "read_snapshot must not be called when connection is None"
    assert isinstance(result, FallbackResult)
    assert result.source == "fallback_cached"


# ---------------------------------------------------------------- #
# both empty -> None
# ---------------------------------------------------------------- #
def test_both_tiers_empty_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_read_snapshot(monkeypatch, [])
    result = serve_fallback(
        connection=_FakeConnection(),
        popularity_cache=_cache([]),
        k=2,
        filters=None,
    )
    assert result is None


def test_no_connection_and_empty_cache_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_read_snapshot(monkeypatch, [("i_1", 1.0)])
    result = serve_fallback(
        connection=None,
        popularity_cache=_cache([]),
        k=2,
        filters=None,
    )
    assert result is None


# ---------------------------------------------------------------- #
# filters are forwarded to the in-memory cache unchanged
# ---------------------------------------------------------------- #
def test_filters_forwarded_to_popularity_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_read_snapshot(monkeypatch, [])
    cache = _cache(
        [
            _snapshot_item("i_match", 1),
            _snapshot_item("i_other", 2),
        ]
    )
    # Both items share category=books; filter to language=en which
    # both have, so the cache returns both. The test asserts the
    # filters were passed, not that the cache filtered (that is the
    # cache's own test).
    result = serve_fallback(
        connection=_FakeConnection(),
        popularity_cache=cache,
        k=10,
        filters={"language": "en"},
    )
    assert isinstance(result, FallbackResult)
    assert result.source == "fallback_cached"
