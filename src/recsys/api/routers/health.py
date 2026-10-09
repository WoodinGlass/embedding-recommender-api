"""Health and readiness endpoints — see ``docs/contracts.md`` § 2.6.

Three endpoints. ``/livez`` is the liveness probe; ``/healthz`` is
its historical alias. ``/readyz`` reports four checks and a
three-state aggregate (ADR-0019).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from recsys.api.readiness import (
    NOT_CONFIGURED,
    aggregate,
    check_db,
    check_index,
    check_redis,
)
from recsys.config.enums import AppEnv
from recsys.config.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/livez", summary="Liveness probe")
async def livez() -> dict[str, str]:
    """Return 200 as long as the process is running.

    Dependency checks live on :func:`readyz`; a liveness probe that
    read the database would restart a healthy pod on a database
    hiccup (ADR-0019).
    """
    return {"status": "alive"}


@router.get("/healthz", summary="Liveness probe (alias)")
async def healthz() -> dict[str, str]:
    """Alias of ``/livez``; kept for the M0 scaffold and the smoke test."""
    return {"status": "alive"}


@router.get("/readyz", summary="Readiness probe")
async def readyz(request: Request) -> JSONResponse:
    """Return the readiness state with a per-dependency breakdown.

    ``db`` and ``index`` are required; ``encoder`` is required in
    prod; ``redis`` is optional. The aggregate is ``ready`` (all
    pass), ``degraded`` (required pass, an optional fails), or
    ``not_ready`` (a required check fails). HTTP is 200 for ready
    and degraded, 503 for not_ready.
    """
    settings = get_settings()
    checks: dict[str, dict[str, object]] = {}

    pool = getattr(request.app.state, "db_pool", None)
    if pool is None or getattr(pool, "closed", True):
        # The pool never opened (startup database unreachable). Every
        # check that needs a connection reports the same category.
        checks["db"] = {"ok": False, "required": True, "error": NOT_CONFIGURED}
        checks["index"] = {"ok": False, "required": True, "error": NOT_CONFIGURED}
    else:
        checks["db"] = check_db(pool)
        if checks["db"].get("ok") is True:
            checks["index"] = check_index(pool)
        else:
            # No point querying index_registry over a broken pool.
            checks["index"] = {
                "ok": False,
                "required": True,
                "error": NOT_CONFIGURED,
            }

    encoder = getattr(request.app.state, "encoder", None)
    required_in_env = settings.app_env is AppEnv.PROD
    encoder_ok = encoder is not None
    checks["encoder"] = {"ok": encoder_ok, "required": required_in_env}
    if not encoder_ok:
        checks["encoder"]["error"] = "artifact_missing_or_unusable"

    redis_url = getattr(settings, "redis_url", "") or ""
    if not redis_url:
        checks["redis"] = {"ok": False, "required": False, "error": NOT_CONFIGURED}
    else:
        checks["redis"] = await check_redis(redis_url)

    status, code = aggregate(checks)
    return JSONResponse(
        status_code=code,
        content={"status": status, "checks": checks},
    )
