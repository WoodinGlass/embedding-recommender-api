"""Application settings — the only place environment variables are read.

Two sentinels are used to detect "not yet configured" in production:
``JWT_SECRET`` and ``USER_ID_HASH_SALT``. Both default to a value that
starts with ``change-me``; the prod guard refuses to boot when either
still carries its sentinel. A sentinel is a placeholder, not a secret;
the noqa comments document that intent for the bandit rules that flag
any string assigned to a ``*_secret`` or ``*_salt`` field.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from recsys.config.enums import AppEnv, Device, IndexBackend, LogFormat, LogLevel

_DEV_JWT_SECRET_SENTINEL = "change-me-before-any-real-deployment"  # noqa: S105

_DEV_USER_ID_HASH_SALT_SENTINEL = "change-me-before-any-real-deployment"

#: Minimum length for a JWT secret in production. The value is a
#: judgment: 32 characters of random data is a reasonable floor for an
#: HMAC-SHA256 key. A shorter secret is refused; a longer one is fine.
_JWT_SECRET_MIN_LENGTH = 32

#: The prefix Argon2id hashes use in PHC format. Repeating the constant
#: here (rather than importing it from ``recsys.api.auth.api_key``)
#: keeps the config layer free of an upward import into the API layer.
_ARGON2_PREFIX = "$argon2"


class Settings(BaseSettings):
    """Environment-driven configuration.

    Loaded from process environment (case-insensitive) and, when present,
    from a ``.env`` file. See ``.env.example`` and ``docs/contracts.md``
    section 3.1 for the full list. Startup fails on invalid values; there
    is no silent fallback.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # application
    # ------------------------------------------------------------------ #
    app_env: AppEnv = AppEnv.DEV
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.JSON
    metrics_enabled: bool = True

    # ------------------------------------------------------------------ #
    # storage
    # ------------------------------------------------------------------ #
    database_url: str = "postgresql://recsys:recsys@postgres:5432/recsys"
    redis_url: str = "redis://redis:6379/0"
    #: Per-call timeout for the cache Redis client. Redis at 100 ms
    #: is a judgment: a call that exceeds it means the cache is not
    #: helping, so it should be treated as a miss. See ADR-0015.
    cache_socket_timeout_seconds: float = Field(default=0.1, gt=0)
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86_400)

    # ------------------------------------------------------------------ #
    # auth (ADR-0013)
    #
    # API_KEYS and API_KEYS_ADMIN are semicolon-separated lists. The
    # separator is ";" and not "," because an Argon2id PHC hash contains
    # commas (in its parameters section: "$argon2id$v=19$m=65536,t=3,p=4$...")
    # and a comma-split would fragment every hash into pieces that no
    # longer look like hashes.
    # ------------------------------------------------------------------ #
    api_keys: str = ""
    api_keys_admin: str = ""
    jwt_secret: str = _DEV_JWT_SECRET_SENTINEL
    jwt_algorithm: str = "HS256"
    jwt_max_age_seconds: int = Field(default=86_400, ge=1)

    # ------------------------------------------------------------------ #
    # rate limiting (ADR-0014)
    #
    # Two-tier limiting: an IP bucket that is always consulted, and a
    # credential bucket consulted when a credential is present. The IP
    # limit is intentionally higher: a single IP can be a NAT serving
    # many users, and the IP bucket exists to bound anonymous or
    # credential-rotating traffic, not to be the primary limit.
    # ------------------------------------------------------------------ #
    rate_limit_per_minute: int = Field(default=600, ge=1)
    rate_limit_ip_per_minute: int = Field(default=5000, ge=1)
    #: Burst window for the token bucket. The bucket capacity is
    #: ``rate * (bucket_seconds / 60)``; 60 s means "a full minute of
    #: tokens may burst at once". See ADR-0014.
    rate_limit_bucket_seconds: float = Field(default=60.0, gt=0)
    instance_count: int = Field(default=1, ge=1)
    # Number of trusted reverse proxies between the client and the API.
    # 0 means the direct peer is the client (no proxy headers trusted).
    # 1 means one trusted proxy (X-Forwarded-For's last entry is the
    # client). Values above 1 follow the same rule for the Nth entry
    # from the right. Never trust a header supplied by an untrusted
    # peer; see ADR-0014.
    trusted_proxy_count: int = Field(default=0, ge=0, le=10)

    # ------------------------------------------------------------------ #
    # circuit breakers (ADR-0015)
    # ------------------------------------------------------------------ #
    redis_breaker_failure_threshold: int = Field(default=5, ge=1)
    redis_breaker_open_seconds: float = Field(default=5.0, gt=0)
    redis_breaker_open_max_seconds: float = Field(default=60.0, gt=0)
    redis_breaker_timeout_seconds: float = Field(default=0.1, gt=0)
    db_breaker_failure_threshold: int = Field(default=5, ge=1)
    db_breaker_open_seconds: float = Field(default=2.0, gt=0)
    db_breaker_open_max_seconds: float = Field(default=30.0, gt=0)
    db_breaker_timeout_seconds: float = Field(default=2.0, gt=0)

    # ------------------------------------------------------------------ #
    # event ingestion (ADR-0018)
    # ------------------------------------------------------------------ #
    event_ts_max_age_seconds: int = Field(default=604_800, ge=1)
    event_ts_max_future_seconds: int = Field(default=300, ge=0)
    user_id_hash_salt: str = _DEV_USER_ID_HASH_SALT_SENTINEL
    user_id_hash_salt_version: int = Field(default=0, ge=0)
    event_retention_days: int = Field(default=365, ge=1)

    # ------------------------------------------------------------------ #
    # readiness (ADR-0019)
    # ------------------------------------------------------------------ #
    readyz_cache_seconds: int = Field(default=5, ge=0, le=60)
    readyz_db_timeout_seconds: float = Field(default=0.5, gt=0)
    readyz_index_timeout_seconds: float = Field(default=0.5, gt=0)
    readyz_redis_timeout_seconds: float = Field(default=0.1, gt=0)

    # ------------------------------------------------------------------ #
    # observability (ADR-0021)
    # ------------------------------------------------------------------ #
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    otel_service_name: str = "recsys-api"
    otel_traces_sampler_arg: float = Field(default=1.0, ge=0.0, le=1.0)

    # ------------------------------------------------------------------ #
    # config pollers (ADR-0022)
    # ------------------------------------------------------------------ #
    hot_config_poll_seconds: int = Field(default=5, ge=1)
    hot_config_path: str = "config/hot.yaml"
    active_index_poll_seconds: int = Field(default=10, ge=1)

    # ------------------------------------------------------------------ #
    # embeddings (M1)
    # ------------------------------------------------------------------ #
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, ge=1, le=1024)
    embedding_onnx_path: str = (
        # Must match what scripts/export_onnx.py produces: the model
        # name with "/" replaced by "__" (see model_slug in that
        # script). Keeping these in sync is what makes
        # `python -m recsys.embeddings.pipeline` work out of the box
        # after `python scripts/export_onnx.py`.
        "artifacts/onnx/sentence-transformers__all-MiniLM-L6-v2"
    )
    device: Device = Device.CPU

    # ------------------------------------------------------------------ #
    # retrieval (M2)
    # ------------------------------------------------------------------ #
    index_backend: IndexBackend = IndexBackend.PGVECTOR
    hnsw_m: int = Field(default=16, ge=2, le=128)
    hnsw_ef_construction: int = Field(default=64, ge=2, le=1024)
    hnsw_ef_search: int = Field(default=100, ge=1, le=1024)
    ann_timeout_ms: int = Field(default=120, ge=1, le=5000)

    # ------------------------------------------------------------------ #
    # experiments (ADR-0017)
    # ------------------------------------------------------------------ #
    experiment_salt_default: str = "recsys-default-v1"

    # ------------------------------------------------------------------ #
    # deployment (ADR-0023)
    # ------------------------------------------------------------------ #
    pre_stop_delay_seconds: int = Field(default=5, ge=0)

    # ------------------------------------------------------------------ #
    # SLO (ADR-0024)
    # ------------------------------------------------------------------ #
    latency_slo_ms: int = Field(default=200, ge=1)

    # ================================================================== #
    # validators
    # ================================================================== #
    @field_validator("hnsw_ef_construction")
    @classmethod
    def _ef_construction_ge_m(cls, v: int, info: ValidationInfo) -> int:
        m = info.data.get("hnsw_m")
        if m is not None and v < m:
            raise ValueError("HNSW_EF_CONSTRUCTION must be >= HNSW_M")
        return v

    @field_validator("redis_breaker_open_max_seconds")
    @classmethod
    def _redis_open_max_ge_open(cls, v: float, info: ValidationInfo) -> float:
        base = info.data.get("redis_breaker_open_seconds")
        if base is not None and v < base:
            raise ValueError("REDIS_BREAKER_OPEN_MAX_SECONDS must be >= REDIS_BREAKER_OPEN_SECONDS")
        return v

    @field_validator("db_breaker_open_max_seconds")
    @classmethod
    def _db_open_max_ge_open(cls, v: float, info: ValidationInfo) -> float:
        base = info.data.get("db_breaker_open_seconds")
        if base is not None and v < base:
            raise ValueError("DB_BREAKER_OPEN_MAX_SECONDS must be >= DB_BREAKER_OPEN_SECONDS")
        return v

    @field_validator("jwt_algorithm")
    @classmethod
    def _jwt_algorithm_allowed(cls, v: str) -> str:
        allowed = {"HS256", "RS256"}
        if v not in allowed:
            raise ValueError(f"JWT_ALGORITHM must be one of {sorted(allowed)}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _prod_guard(self) -> Settings:
        """Refuse to boot in prod with a dev-shaped configuration.

        Every check here is a failure mode that would otherwise surface
        on the first request that needs the value. Failing at startup
        makes the misconfiguration a container-start failure, which an
        orchestrator can act on.
        """
        if self.app_env is not AppEnv.PROD:
            return self

        if not self.api_keys.strip():
            raise ValueError("API_KEYS must be set when APP_ENV=prod")

        # A plaintext key in production is a shared secret in a config
        # file that operators copy around. Every entry must be an
        # Argon2id hash; the development path's plaintext entries are
        # refused.
        for key in self.api_key_set:
            if not key.startswith(_ARGON2_PREFIX):
                raise ValueError(
                    "API_KEYS entries must be Argon2id hashes in prod; "
                    "the plaintext path is dev-only "
                    "(see docs/ops.md section 8.1)"
                )

        if self.jwt_secret.startswith("change-me"):
            raise ValueError("JWT_SECRET must not be the default in prod")
        if len(self.jwt_secret) < _JWT_SECRET_MIN_LENGTH:
            raise ValueError(
                f"JWT_SECRET must be at least {_JWT_SECRET_MIN_LENGTH} characters in prod"
            )

        if self.user_id_hash_salt.startswith("change-me"):
            raise ValueError("USER_ID_HASH_SALT must not be the default in prod")
        if self.user_id_hash_salt_version < 1:
            raise ValueError(
                "USER_ID_HASH_SALT_VERSION must be >= 1 in prod "
                "(a rotation always bumps the version)"
            )

        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL must be a postgresql:// URL when APP_ENV=prod")
        if not self.redis_url.startswith(("redis://", "rediss://")):
            raise ValueError("REDIS_URL must be a redis:// or rediss:// URL when APP_ENV=prod")

        return self

    # ================================================================== #
    # derived values
    # ================================================================== #
    @property
    def api_key_set(self) -> frozenset[str]:
        """Parsed, trimmed set of read API keys (hashes or plaintext).

        Entries are separated by ``;`` (see the field comment above): an
        Argon2id PHC hash contains commas, so a comma-separated list
        cannot hold hashes without fragmenting them.
        """
        return frozenset(k.strip() for k in self.api_keys.split(";") if k.strip())

    @property
    def api_key_admin_set(self) -> frozenset[str]:
        """Parsed, trimmed set of admin API keys (hashes or plaintext)."""
        return frozenset(k.strip() for k in self.api_keys_admin.split(";") if k.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so import-time callers do not re-read the environment. Tests
    reset with ``get_settings.cache_clear()``.
    """
    return Settings()
