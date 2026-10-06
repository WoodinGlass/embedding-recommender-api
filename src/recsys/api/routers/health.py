"""Health and readiness endpoints — see ``docs/contracts.md`` § 2."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict[str, str]:
    """Return 200 as long as the process is running.

    Dependency checks live on :func:`readyz`.
    """
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
async def readyz() -> dict[str, object]:
    """Return readiness state.

    The M0 scaffold has no live dependencies yet, so this reports ready. From
    M3 onwards, the DB and active index checks will gate the response and
    return 503 when the instance must be drained.
    """
    return {"status": "ready", "checks": {}}
