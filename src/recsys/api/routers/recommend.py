"""Recommendation endpoints.

Stubs return ``503`` with a clear milestone message. Implementation lands in
M3 (Production API). The schemas are already final; only the handlers change.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from recsys.api.schemas.recommend import RecommendRequest, RecommendResponse

router = APIRouter(prefix="/v1", tags=["recommend"])

_NOT_YET = "Recommendation endpoint is implemented in milestone M3."


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Top-k recommendations with metadata filters",
)
async def recommend(_request: RecommendRequest) -> RecommendResponse:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=_NOT_YET,
    )


@router.get(
    "/items/{item_id}/similar",
    response_model=RecommendResponse,
    summary="Item-to-item similarity",
)
async def similar(item_id: str) -> RecommendResponse:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Item-to-item similarity for {item_id!r} lands in M3.",
    )
