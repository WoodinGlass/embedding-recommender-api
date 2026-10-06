"""Configuration enums.

Every selector is a :class:`enum.StrEnum` so an invalid value aborts startup
rather than silently falling back. See ``docs/contracts.md`` § 3 for the
config contract.
"""

from __future__ import annotations

from enum import StrEnum


class AppEnv(StrEnum):
    """Deployment environment. ``prod`` enables extra startup guards."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(StrEnum):
    JSON = "json"
    CONSOLE = "console"


class IndexBackend(StrEnum):
    """Vector index backend. Adding a third requires an ADR."""

    PGVECTOR = "pgvector"
    FAISS = "faiss"


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
