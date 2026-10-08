"""Unit tests for recsys.rate_limit.keys (pure functions, ADR-0014)."""

from __future__ import annotations

import pytest

from recsys.rate_limit.keys import (
    credential_bucket_id,
    ip_bucket_id,
    is_valid_ip,
)


# ---------------------------------------------------------------- #
# is_valid_ip
# ---------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    ["1.2.3.4", "255.255.255.255", "203.0.113.1", "::1", "2001:db8::1", "fe80::1"],
)
def test_is_valid_ip_accepts_real_addresses(value: str) -> None:
    assert is_valid_ip(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not an ip",
        "1.2.3.4.5",
        "http://example.com",
        "1,2,3,4",
        "9" * 100,  # over the 45-char structural bound
    ],
)
def test_is_valid_ip_rejects_obvious_non_addresses(value: str) -> None:
    assert is_valid_ip(value) is False


def test_is_valid_ip_is_structural_not_authoritative() -> None:
    """A hex-only string passes the IPv6 shape check even though it is not
    a routable address. The docstring documents this: the check is a
    structural filter, not a network parser."""
    assert is_valid_ip("abc") is True


# ---------------------------------------------------------------- #
# ip_bucket_id
# ---------------------------------------------------------------- #
def test_ip_bucket_id_zero_trusted_uses_peer() -> None:
    assert (
        ip_bucket_id(
            peer_ip="203.0.113.5",
            forwarded_for=["1.2.3.4"],
            trusted_proxy_count=0,
        )
        == "203.0.113.5"
    )


def test_ip_bucket_id_zero_trusted_ignores_forwarded_header() -> None:
    """With no trusted proxies, a forwarded header from an untrusted peer
    must be ignored entirely."""
    assert (
        ip_bucket_id(
            peer_ip="203.0.113.5",
            forwarded_for=["1.2.3.4", "5.6.7.8"],
            trusted_proxy_count=0,
        )
        == "203.0.113.5"
    )


def test_ip_bucket_id_takes_nth_from_right() -> None:
    forwarded = ["198.51.100.7", "10.0.0.2"]
    # trusted_proxy_count=1 -> rightmost entry was written by the last proxy
    assert (
        ip_bucket_id(peer_ip="10.0.0.1", forwarded_for=forwarded, trusted_proxy_count=1)
        == "10.0.0.2"
    )
    # trusted_proxy_count=2 -> the entry before that
    assert (
        ip_bucket_id(peer_ip="10.0.0.1", forwarded_for=forwarded, trusted_proxy_count=2)
        == "198.51.100.7"
    )


def test_ip_bucket_id_chain_shorter_than_trusted_returns_unknown() -> None:
    assert (
        ip_bucket_id(
            peer_ip="10.0.0.1",
            forwarded_for=["1.2.3.4"],
            trusted_proxy_count=5,
        )
        == "unknown"
    )


def test_ip_bucket_id_invalid_candidate_returns_unknown() -> None:
    assert (
        ip_bucket_id(
            peer_ip="10.0.0.1",
            forwarded_for=["not-an-ip"],
            trusted_proxy_count=1,
        )
        == "unknown"
    )


def test_ip_bucket_id_peer_none_returns_unknown() -> None:
    assert ip_bucket_id(peer_ip=None, forwarded_for=[], trusted_proxy_count=0) == "unknown"


def test_ip_bucket_id_peer_invalid_returns_unknown() -> None:
    assert ip_bucket_id(peer_ip="garbage", forwarded_for=[], trusted_proxy_count=0) == "unknown"


# ---------------------------------------------------------------- #
# credential_bucket_id
# ---------------------------------------------------------------- #
def test_credential_bucket_id_none_returns_none() -> None:
    assert credential_bucket_id(None, salt="s") is None


@pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
def test_credential_bucket_id_empty_returns_none(raw: str) -> None:
    assert credential_bucket_id(raw, salt="s") is None


def test_credential_bucket_id_is_32_lowercase_hex() -> None:
    bucket = credential_bucket_id("some-key", salt="s")
    assert bucket is not None
    assert len(bucket) == 32
    assert all(c in "0123456789abcdef" for c in bucket)


def test_credential_bucket_id_is_deterministic() -> None:
    assert credential_bucket_id("k", salt="s") == credential_bucket_id("k", salt="s")


def test_credential_bucket_id_salt_changes_hash() -> None:
    assert credential_bucket_id("k", salt="s1") != credential_bucket_id("k", salt="s2")


def test_credential_bucket_id_different_credentials_differ() -> None:
    assert credential_bucket_id("k1", salt="s") != credential_bucket_id("k2", salt="s")


def test_credential_bucket_id_truncates_long_input() -> None:
    """Inputs longer than _MAX_CREDENTIAL_LENGTH (4096) are truncated before
    hashing, so two credentials that share the first 4096 characters collide
    by design."""
    long_a = "a" * 10_000
    long_b = "a" * 4096 + "x" * 10
    assert credential_bucket_id(long_a, salt="s") == credential_bucket_id(long_b, salt="s")


def test_credential_bucket_id_does_not_leak_raw_credential() -> None:
    bucket = credential_bucket_id("very-secret-key", salt="s")
    assert bucket is not None
    assert "very-secret-key" not in bucket
