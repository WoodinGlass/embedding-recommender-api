"""Application settings — the only place environment variables are read."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from recsys.config.enums import AppEnv, Device, IndexBackend, LogFormat, LogLevel

# Sentinel used as the default for ``jwt_secret`` in dev only. The prod guard
# below refuses to boot when the secret still starts with "change-me". This is
# a well-known placeholder, not a secret; the noqa documents that intent.
_DEV_JWT_SECRET_SENTINEL = "change-me-before-any-real-deployment"  # noqa: S105


class Settings(BaseSettings):
    """Environment-driven configuration.

    Loaded from process environment (case-insensitive) and, when present, from
    a ``.env`` file. See ``.env.example`` for the full list. Startup fails on
    invalid values — there is no silent fallback.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # application ---------------------------------------------------------
    app_env: AppEnv = AppEnv.DEV
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.JSON

    # storage -------------------------------------------------------------
    database_url: str = "postgresql://recsys:recsys@postgres:5432/recsys"
    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86_400)

    # auth ----------------------------------------------------------------
    api_keys: str = ""
    jwt_secret: str = _DEV_JWT_SECRET_SENTINEL
    jwt_algorithm: str = "HS256"
    rate_limit_per_minute: int = Field(default=600, ge=1)

    # embeddings ----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, ge=1, le=1024)
    embedding_onnx_path: str = "artifacts/onnx/all-MiniLM-L6-v2"
    device: Device = Device.CPU

    # retrieval -----------------------------------------------------------
    index_backend: IndexBackend = IndexBackend.PGVECTOR
    hnsw_m: int = Field(default=16, ge=2, le=128)
    hnsw_ef_construction: int = Field(default=64, ge=2, le=1024)
    hnsw_ef_search: int = Field(default=100, ge=1, le=1024)
    ann_timeout_ms: int = Field(default=120, ge=1, le=5000)

    # experiments ---------------------------------------------------------
    experiment_salt_default: str = "recsys-default-v1"

    # observability -------------------------------------------------------
    otel_exporter_otlp_endpoint: str = "http://otel-collector:4317"
    otel_service_name: str = "recsys-api"
    metrics_enabled: bool = True

    # ---------------------------------------------------------------------
    # validators
    # ---------------------------------------------------------------------
    @field_validator("hnsw_ef_construction")
    @classmethod
    def _ef_construction_ge_m(cls, v: int, info: ValidationInfo) -> int:
        m = info.data.get("hnsw_m")
        if m is not None and v < m:
            raise ValueError("HNSW_EF_CONSTRUCTION must be >= HNSW_M")
        return v

    @model_validator(mode="after")
    def _prod_guard(self) -> Settings:
        """Refuse to boot in prod with dev defaults for secrets."""
        if self.app_env is AppEnv.PROD:
            if not self.api_keys.strip():
                raise ValueError("API_KEYS must be set when APP_ENV=prod")
            if self.jwt_secret.startswith("change-me"):
                raise ValueError("JWT_SECRET must not be the default in prod")
        return self

    # ---------------------------------------------------------------------
    # derived values
    # ---------------------------------------------------------------------
    @property
    def api_key_set(self) -> frozenset[str]:
        """Parsed, trimmed set of accepted API keys."""
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so import-time callers do not re-read the environment. Tests reset
    with ``get_settings.cache_clear()``.
    """
    return Settings()
