"""Churn scoring endpoint — extension, lands in M7."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from recsys.api.deps import PrincipalDep
from recsys.api.schemas.churn import ChurnScoreRequest, ChurnScoreResponse

router = APIRouter(prefix="/v1/churn", tags=["churn"])


@router.post("/score", response_model=ChurnScoreResponse)
async def score(_request: ChurnScoreRequest, principal: PrincipalDep) -> ChurnScoreResponse:
    del principal  # auth only; the handler lands in M7
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Churn scoring lands in M7.",
    )
