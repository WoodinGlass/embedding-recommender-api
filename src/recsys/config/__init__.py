"""Configuration package — see ``docs/contracts.md`` § 3."""

from recsys.config.enums import AppEnv, Device, IndexBackend, LogFormat, LogLevel
from recsys.config.settings import Settings, get_settings

__all__ = [
    "AppEnv",
    "Device",
    "IndexBackend",
    "LogFormat",
    "LogLevel",
    "Settings",
    "get_settings",
]
