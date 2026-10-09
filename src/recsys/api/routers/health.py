"""Health and readiness endpoints — see ``docs/contracts.md`` § 2.6."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from recsys.config.enums import AppEnv
from recsys.config.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Return 200 as long as the process is running.

    Dependency checks live on :func:`readyz`.
    """
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
async def readyz(request: Request) -> JSONResponse:
    """Return readiness state with a per-dependency breakdown.

    The response shape is fixed by ``docs/contracts.md`` § 2.6: a
    top-level ``status`` (``ready`` / ``degraded`` / ``not_ready``)
    and a ``checks`` map. This handler reports the encoder only; the
    database, the active index, and Redis are added in M3.7 along
    with the readiness cache (ADR-0019).

    The encoder check is the first one to land because its failure
    mode is silent: without an encoder the handler serves popular
    items and the relevance drop is invisible until someone reads a
    dashboard. In prod the encoder is required at startup (the app
    factory refuses to boot without it), so `required` here is a
    reminder of that contract rather than an independent gate. In
    dev and test, missing is allowed and the check reports it as
    `required: false` so an orchestrator does not drain the
    instance.
    """
    settings = get_settings()
    encoder = getattr(request.app.state, "encoder", None)
    required_in_env = settings.app_env is AppEnv.PROD
    encoder_ok = encoder is not None

    checks: dict[str, dict[str, object]] = {
        "encoder": {
            "ok": encoder_ok,
            "required": required_in_env,
        }
    }
    if not encoder_ok:
        checks["encoder"]["error"] = "artifact_missing_or_unusable"

    # The aggregate status: ready when every required check is ok.
    # Because the only check today is required only in prod, a dev
    # instance with no encoder is still ready.
    required_checks = [c for c in checks.values() if c.get("required") is True]
    if all(c.get("ok") is True for c in required_checks):
        status = "ready"
        code = 200
    else:
        status = "not_ready"
        code = 503

    return JSONResponse(
        status_code=code,
        content={"status": status, "checks": checks},
    )
