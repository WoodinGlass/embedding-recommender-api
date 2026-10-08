"""Rate limiting (ADR-0014).

A token bucket in Redis, executed as one Lua script per request, with a
per-instance in-memory fallback when Redis is unavailable. The public
surface is :class:`TokenBucketLimiter` and its decision value.
"""

from recsys.rate_limit.keys import credential_bucket_id, ip_bucket_id
from recsys.rate_limit.limiter import RateLimitDecision, TokenBucketLimiter

__all__ = [
    "RateLimitDecision",
    "TokenBucketLimiter",
    "credential_bucket_id",
    "ip_bucket_id",
]
