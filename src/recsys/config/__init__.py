"""Configuration package.

Two layers:

- ``settings`` — boot-time, environment-driven (frozen for the life of
  the process). Changing it needs a restart.
- ``hot`` — file-driven, reloadable within ``HOT_CONFIG_POLL_SECONDS``
  (ADR-0022). Not for secrets; the file is committed.
"""

from recsys.config.hot import (
    CacheSection,
    HotConfig,
    HotConfigError,
    HotConfigStore,
    RateLimitSection,
    RerankSection,
    default_hot_config,
    load_hot_config,
)
from recsys.config.settings import Settings, get_settings

__all__ = [
    "CacheSection",
    "HotConfig",
    "HotConfigError",
    "HotConfigStore",
    "RateLimitSection",
    "RerankSection",
    "Settings",
    "default_hot_config",
    "get_settings",
    "load_hot_config",
]
