"""Key derivation for the two rate-limit tiers (ADR-0014).

Two functions, each producing the "bucket id" that
:class:`recsys.rate_limit.limiter.TokenBucketLimiter` keys on:

- :func:`ip_bucket_id` extracts the client IP from the request, honoring
  the configured number of trusted proxies. It never trusts a forwarded
  header from an untrusted peer.
- :func:`credential_bucket_id` produces a stable, non-reversible
  identifier for whatever credential the request carries. It uses
  HMAC-SHA256 with the same salt as the user-id hasher (ADR-0018) so a
  credential never appears in a Redis key or a log line, and the hash
  is not dictionary-reversible against a predictable credential.

Both functions are pure: they take their inputs as arguments, do not
read config, do not log, and do not touch Redis.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from recsys.security.hashing import hmac_sha256

#: A conservative IPv4/IPv6 shape check. A malformed value in the header
#: is replaced by a fixed sentinel (``"unknown"``) rather than being
#: truncated or repaired: an unparsable address is a signal that the
#: proxy chain is misconfigured, and every such request landing in one
#: bucket is easier to see than a bucket per malformed value.
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_IPV6_RE = re.compile(r"^[0-9a-fA-F:]+$")

#: Credentials longer than this are truncated before hashing. The bound
#: exists so a caller cannot force a large allocation by presenting an
#: arbitrarily long credential; 4 KiB is far above any real key or JWT.
_MAX_CREDENTIAL_LENGTH = 4096


def is_valid_ip(value: str) -> bool:
    """Return True when ``value`` looks like an IPv4 or IPv6 address.

    The check is structural, not authoritative: it is enough to reject
    a header that obviously is not an address (an empty string, a token
    that contains a comma, a URL). Parsing with the ``ipaddress`` module
    would be stricter, but this function is on the hot path and the
    value it accepts is only used as a Redis key, not as a network
    address.
    """
    if not value or len(value) > 45:
        return False
    return bool(_IPV4_RE.match(value) or _IPV6_RE.match(value))


def ip_bucket_id(
    *,
    peer_ip: str | None,
    forwarded_for: Sequence[str],
    trusted_proxy_count: int,
) -> str:
    """Return the client IP to key the IP rate-limit bucket on.

    ``forwarded_for`` is the parsed ``X-Forwarded-For`` list (the raw
    header split on commas, whitespace stripped). ``trusted_proxy_count``
    is the number of reverse proxies between the client and this process.

    - When ``trusted_proxy_count == 0``, the header is ignored (an
      untrusted peer could have set it) and the direct peer's address is
      used.
    - When ``trusted_proxy_count == N > 0``, the value is the Nth entry
      from the *right* of ``X-Forwarded-For``: each trusted proxy
      appends the address it saw, so the rightmost entry was written by
      the last trusted proxy and the Nth-from-right is the first
      untrusted one.
    - When the header is shorter than ``trusted_proxy_count``, or its
      chosen entry is not a valid address, the result is the sentinel
      ``"unknown"``. Every such request lands in one bucket, which is
      the honest behavior: the proxy chain is misconfigured, and
      lumping the misconfigured requests together is more visible than
      spreading them out.

    ``peer_ip`` may be ``None`` (Starlette returns ``None`` for
    ``request.client`` when the ASGI server did not report a peer, e.g.
    in a unit test that fabricates a scope).
    """
    if trusted_proxy_count == 0:
        return peer_ip if peer_ip and is_valid_ip(peer_ip) else "unknown"

    chain = list(forwarded_for)
    if len(chain) < trusted_proxy_count:
        return "unknown"
    # Nth from the right, 1-based.
    candidate = chain[-trusted_proxy_count]
    return candidate if is_valid_ip(candidate) else "unknown"


def credential_bucket_id(raw_credential: str | None, *, salt: str) -> str | None:
    """Return the bucket id for a presented credential, or ``None``.

    ``None`` means "no credential to key on"; the caller then applies
    only the IP bucket. A non-empty credential is HMAC-SHA256 hashed
    with ``salt`` and truncated to 32 hex characters, which is what the
    Redis key holds. The raw credential never leaves this function.

    The salt is the same secret the user-id hasher uses (ADR-0018), so a
    credential whose input space is small is not dictionary-reversible
    from the hash alone. Truncating to 32 hex characters halves the
    key-name memory without meaningfully weakening the identifier: the
    bucket exists to meter, not to authenticate, and the
    authentication layer (ADR-0013) is what validates the credential.
    """
    if raw_credential is None:
        return None
    candidate = raw_credential.strip()
    if not candidate:
        return None
    if len(candidate) > _MAX_CREDENTIAL_LENGTH:
        candidate = candidate[:_MAX_CREDENTIAL_LENGTH]
    digest = hmac_sha256(salt, candidate)
    return digest[:32]
