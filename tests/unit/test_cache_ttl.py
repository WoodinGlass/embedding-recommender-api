"""Unit tests for jittered TTL and the negative sentinel (ADR-0015)."""

from __future__ import annotations

from random import Random

import pytest

from recsys.cache.ttl import NEGATIVE_SENTINEL, jittered_ttl


# ---------------------------------------------------------------- #
# sentinel
# ---------------------------------------------------------------- #
def test_sentinel_is_bytes() -> None:
    assert isinstance(NEGATIVE_SENTINEL, bytes)


def test_sentinel_starts_with_nul() -> None:
    """The leading NUL is what makes it collision-proof against a
    serialized payload (a real value starts with ``[``)."""
    assert NEGATIVE_SENTINEL[:1] == b"\x00"


def test_sentinel_is_not_a_falsey_value() -> None:
    assert NEGATIVE_SENTINEL != b""
    assert NEGATIVE_SENTINEL != b"0"
    assert NEGATIVE_SENTINEL != b"null"
    assert NEGATIVE_SENTINEL != b"[]"


# ---------------------------------------------------------------- #
# jittered_ttl
# ---------------------------------------------------------------- #
def test_returns_int() -> None:
    assert isinstance(jittered_ttl(300), int)


def test_never_below_one() -> None:
    # base 1 with the worst case (0.9) rounds toward 0; the max(1, ...)
    # in the function must clamp it.
    for seed in range(50):
        assert jittered_ttl(1, rng=Random(seed)) >= 1


def test_within_jitter_band() -> None:
    base = 1000
    for seed in range(200):
        value = jittered_ttl(base, rng=Random(seed))
        assert int(base * 0.9) <= value <= int(base * 1.1), value


def test_deterministic_with_same_seed() -> None:
    a = jittered_ttl(300, rng=Random(42))
    b = jittered_ttl(300, rng=Random(42))
    assert a == b


def test_actually_jitters() -> None:
    """Across many RNG seeds, the function must produce more than one
    distinct TTL for the same base. If it did not, jitter is a no-op
    and the stampede it exists to prevent is not prevented."""
    values = {jittered_ttl(300, rng=Random(seed)) for seed in range(100)}
    assert len(values) > 5, f"jitter produced only {len(values)} distinct values"


def test_zero_base_raises() -> None:
    with pytest.raises(ValueError, match="base_seconds"):
        jittered_ttl(0)


def test_negative_base_raises() -> None:
    with pytest.raises(ValueError, match="base_seconds"):
        jittered_ttl(-5)


def test_default_rng_is_independent_across_calls() -> None:
    """Two calls without an explicit rng use a fresh source each time;
    the sequence is not tied to module state, so a caller cannot
    observe another caller's draws."""
    values = {jittered_ttl(300) for _ in range(100)}
    # Not a distribution test — just "more than one value" to prove
    # the default path is not accidentally pinned to a constant.
    assert len(values) > 1
