"""Hot config loader (ADR-0022).

Reads ``config/hot.yaml`` at startup and re-reads it on change. A
malformed change is logged at ERROR and the previous value stays in
effect; the process does not crash. The file is committed and must not
contain secrets: a field whose name contains a secret marker is
rejected at load time.

Design choices worth stating:

- **Frozen dataclasses, not pydantic.** The schema is small and lives
  in a file the project controls; pydantic would add a dependency to a
  layer that does not need it.
- **The path is a parameter, not a module constant.** Nothing is read
  at import time. Tests pass a tmp file; the app factory passes
  ``Settings.hot_config_path``.
- **No background thread inside the store.** ``reload_if_changed()``
  is explicit, so the caller (the app lifespan) decides how often to
  call it. This keeps the store synchronous and testable without
  ``time.sleep``-based flakiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

#: Only this schema version is accepted. A file with a different value
#: is rejected; a migration is a code change, not a silent upgrade.
SUPPORTED_SCHEMA_VERSION: Final[int] = 1

#: A field whose name contains any of these substrings is rejected at
#: load time. The list is deliberately broad: it is cheaper to reject a
#: legitimate field and rename it than to leak a secret into a log line.
SECRET_MARKERS: Final[frozenset[str]] = frozenset({"secret", "password", "token", "key", "salt"})


def _log() -> Any:
    """Return the structlog logger.

    Called from methods, not at module import: importing
    ``recsys.monitoring.logging`` at module level would create a circular
    import (that module imports ``recsys.config.enums``, which triggers
    this package's ``__init__``, which imports this module).
    """
    from recsys.monitoring.logging import get_logger

    return get_logger(__name__)


# ------------------------------------------------------------------ #
# sections
# ------------------------------------------------------------------ #
@dataclass(frozen=True)
class RateLimitSection:
    recommend_per_minute: int
    events_per_minute: int
    admin_per_minute: int

    def as_class_map(self) -> dict[str, int]:
        """Map the middleware's class names to their per-minute limits."""
        return {
            "recommend": self.recommend_per_minute,
            "events": self.events_per_minute,
            "admin": self.admin_per_minute,
        }


@dataclass(frozen=True)
class RerankSection:
    w_sim: float
    w_pop: float
    w_rec: float
    mmr_lambda: float
    mmr_window: int
    mmr_min_k: int
    candidate_multiplier: int
    recency_half_life_days: int
    enable_mmr: bool

    def to_rerank_config(self) -> Any:
        """Return a :class:`recsys.retrieval.rerank.RerankConfig`.

        The import is lazy, not at module level: ``rerank.py`` imports
        numpy, and ``hot.py`` is loaded at app startup on an image that
        does not install the ``[inference]`` extra. A module-level
        import here would break that image at import time. The method is
        only called from a caller that already needs the re-ranker (and
        therefore numpy), so the lazy import costs nothing there.

        The return type is annotated ``Any`` for the same reason: naming
        ``RerankConfig`` in a signature would force the import at module
        load. The runtime value is a ``RerankConfig``; the field set
        matches and ``RerankConfig.__post_init__`` re-validates, so a
        drift between the two dataclasses is caught at the first call,
        not silently.
        """
        from recsys.retrieval.rerank import RerankConfig

        return RerankConfig(
            w_sim=self.w_sim,
            w_pop=self.w_pop,
            w_rec=self.w_rec,
            mmr_lambda=self.mmr_lambda,
            mmr_window=self.mmr_window,
            mmr_min_k=self.mmr_min_k,
            candidate_multiplier=self.candidate_multiplier,
            recency_half_life_days=self.recency_half_life_days,
            enable_mmr=self.enable_mmr,
        )


@dataclass(frozen=True)
class CacheSection:
    recommend_ttl_seconds: int
    similar_ttl_seconds: int
    negative_ttl_seconds: int
    #: Number of candidates the cache stores (ADR-0015 amendment:
    #: k_max, not k). Sliced to the request's k at serve time.
    cache_k_max: int


@dataclass(frozen=True)
class HotConfig:
    schema_version: int
    rate_limit: RateLimitSection
    rerank: RerankSection
    cache: CacheSection


# ------------------------------------------------------------------ #
# default (used when the file is missing outside prod)
# ------------------------------------------------------------------ #
_DEFAULT = HotConfig(
    schema_version=SUPPORTED_SCHEMA_VERSION,
    rate_limit=RateLimitSection(
        recommend_per_minute=600,
        events_per_minute=6000,
        admin_per_minute=60,
    ),
    rerank=RerankSection(
        w_sim=0.7,
        w_pop=0.2,
        w_rec=0.1,
        mmr_lambda=0.7,
        mmr_window=50,
        mmr_min_k=10,
        candidate_multiplier=4,
        recency_half_life_days=90,
        enable_mmr=False,
    ),
    cache=CacheSection(
        cache_k_max=100,
        recommend_ttl_seconds=300,
        similar_ttl_seconds=600,
        negative_ttl_seconds=30,
    ),
)


def default_hot_config() -> HotConfig:
    """Return the built-in default.

    Used when the file is missing and ``APP_ENV`` is not prod. In prod,
    ``create_app`` refuses to start instead: a missing hot config in prod
    is a misconfiguration, not a fallback situation.
    """
    return _DEFAULT


# ------------------------------------------------------------------ #
# loading
# ------------------------------------------------------------------ #
class HotConfigError(ValueError):
    """Raised when the hot config file is missing, malformed, or unsafe."""


def _reject_secret_fields(node: Any, path: str = "") -> None:
    """Walk a parsed YAML tree; raise when a key looks like a secret."""
    if isinstance(node, dict):
        for k, v in node.items():
            key_str = str(k).lower()
            here = f"{path}.{k}" if path else str(k)
            if any(marker in key_str for marker in SECRET_MARKERS):
                raise HotConfigError(
                    f"hot config field {here!r} looks like a secret; "
                    "hot.yaml is committed and must not carry secrets"
                )
            _reject_secret_fields(v, here)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _reject_secret_fields(item, f"{path}[{i}]")


def _section(node: Any, name: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise HotConfigError(f"hot config section {name!r} must be a mapping")
    return node


def _int_field(
    node: dict[str, Any], name: str, *, minimum: int = 0, default: int | None = None
) -> int:
    if name not in node:
        if default is None:
            raise HotConfigError(f"hot config field {name!r} is required")
        return default
    value = node[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise HotConfigError(
            f"hot config field {name!r} must be an int, got {type(value).__name__}"
        )
    if value < minimum:
        raise HotConfigError(f"hot config field {name!r} must be >= {minimum}, got {value}")
    return value


def _float_field(node: dict[str, Any], name: str, *, minimum: float = 0.0) -> float:
    if name not in node:
        raise HotConfigError(f"hot config field {name!r} is required")
    value = node[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HotConfigError(f"hot config field {name!r} must be a number")
    fvalue = float(value)
    if fvalue < minimum:
        raise HotConfigError(f"hot config field {name!r} must be >= {minimum}, got {fvalue}")
    return fvalue


def _bool_field(node: dict[str, Any], name: str) -> bool:
    """Return a required boolean field.

    ``bool`` is checked before ``int`` because ``True`` is an ``int`` in
    Python; without the explicit check a YAML ``1`` would silently pass
    as ``True`` for a flag field.
    """
    if name not in node:
        raise HotConfigError(f"hot config field {name!r} is required")
    value = node[name]
    if not isinstance(value, bool):
        raise HotConfigError(
            f"hot config field {name!r} must be a bool, got {type(value).__name__}"
        )
    return value


def load_hot_config(path: Path) -> HotConfig:
    """Read, validate, and return the hot config. Raises HotConfigError.

    The function is pure with respect to the process: it opens one
    file, validates it, and returns a value. Nothing global is
    mutated, so a caller that wants to keep the previous value on
    failure can simply not assign the result.
    """
    if not path.is_file():
        raise HotConfigError(f"hot config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HotConfigError(f"hot config is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise HotConfigError("hot config root must be a mapping")

    _reject_secret_fields(raw)

    schema_version = raw.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise HotConfigError(
            f"hot config schema_version must be {SUPPORTED_SCHEMA_VERSION}, got {schema_version!r}"
        )

    rl = _section(raw.get("rate_limit", {}), "rate_limit")
    rr = _section(raw.get("rerank", {}), "rerank")
    ch = _section(raw.get("cache", {}), "cache")

    return HotConfig(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        rate_limit=RateLimitSection(
            recommend_per_minute=_int_field(rl, "recommend_per_minute", minimum=1),
            events_per_minute=_int_field(rl, "events_per_minute", minimum=1),
            admin_per_minute=_int_field(rl, "admin_per_minute", minimum=1),
        ),
        rerank=RerankSection(
            w_sim=_float_field(rr, "w_sim"),
            w_pop=_float_field(rr, "w_pop"),
            w_rec=_float_field(rr, "w_rec"),
            mmr_lambda=_float_field(rr, "mmr_lambda"),
            mmr_window=_int_field(rr, "mmr_window", minimum=1),
            mmr_min_k=_int_field(rr, "mmr_min_k", minimum=1),
            candidate_multiplier=_int_field(rr, "candidate_multiplier", minimum=1),
            recency_half_life_days=_int_field(rr, "recency_half_life_days", minimum=1),
            enable_mmr=_bool_field(rr, "enable_mmr"),
        ),
        cache=CacheSection(
            cache_k_max=_int_field(ch, "cache_k_max", minimum=1, default=100),
            recommend_ttl_seconds=_int_field(ch, "recommend_ttl_seconds", minimum=0),
            similar_ttl_seconds=_int_field(ch, "similar_ttl_seconds", minimum=0),
            negative_ttl_seconds=_int_field(ch, "negative_ttl_seconds", minimum=0),
        ),
    )


# ------------------------------------------------------------------ #
# store
# ------------------------------------------------------------------ #
class HotConfigStore:
    """Holds the current hot config; ``reload_if_changed()`` swaps it.

    A failed reload keeps the previous value and logs the reason; the
    process does not crash. The caller decides how often to call
    ``reload_if_changed()`` (the app lifespan schedules it every
    ``HOT_CONFIG_POLL_SECONDS``).
    """

    def __init__(self, path: Path, *, initial: HotConfig) -> None:
        self._path = path
        self._current = initial
        self._mtime_ns = self._safe_mtime()

    @classmethod
    def from_path(cls, path: Path) -> HotConfigStore:
        """Load the file immediately; raise if the first load fails.

        A first-load failure is a startup failure: the process has no
        previous value to fall back on.
        """
        config = load_hot_config(path)
        return cls(path, initial=config)

    def get(self) -> HotConfig:
        return self._current

    def reload_if_changed(self) -> bool:
        """Reload the file if its mtime changed. Returns True on a swap."""
        mtime = self._safe_mtime()
        if mtime is None or mtime == self._mtime_ns:
            return False
        try:
            new_config = load_hot_config(self._path)
        except HotConfigError as exc:
            _log().error(
                "hot_config.reload_failed",
                path=str(self._path),
                error=str(exc),
            )
            # Remember the bad mtime so we do not retry the same broken
            # file on every poll.
            self._mtime_ns = mtime
            return False
        self._current = new_config
        self._mtime_ns = mtime
        _log().info("hot_config.reloaded", path=str(self._path))
        return True

    def _safe_mtime(self) -> int | None:
        try:
            return self._path.stat().st_mtime_ns
        except OSError:
            return None
