from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.services.claim_service import get_claims, get_claim
from app.schemas.claim import ClaimListResponse, ClaimResponse

router = APIRouter(prefix="/claims", tags=["claims"])


@router.get("", response_model=ClaimListResponse)
async def list_claims(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    risk_level: str | None = Query(None, description="Filter by HIGH, MEDIUM, LOW"),
    payer_id: str | None = Query(None),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: int = Query(-1, description="-1 for desc, 1 for asc"),
    session: AsyncSession = Depends(get_db),
):
    claims, total = await get_claims(
        session,
        skip=skip, limit=limit,
        risk_level=risk_level, payer_id=payer_id,
        sort_by=sort_by, sort_order=sort_order,
    )
    return ClaimListResponse(claims=claims, total=total, skip=skip, limit=limit)


@router.get("/{claim_id}", response_model=ClaimResponse)
async def get_claim_detail(claim_id: str, session: AsyncSession = Depends(get_db)):
    claim = await get_claim(session, claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim
