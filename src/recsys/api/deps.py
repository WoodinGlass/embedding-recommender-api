"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from recsys.config.settings import Settings, get_settings


def get_request_id(request: Request) -> str:
    """Return the request_id assigned by ``RequestContextMiddleware``."""
    return str(getattr(request.state, "request_id", ""))


SettingsDep = Annotated[Settings, Depends(get_settings)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
