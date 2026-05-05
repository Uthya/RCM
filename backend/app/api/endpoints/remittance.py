from fastapi import APIRouter, HTTPException, Query

from app.services.remittance_service import get_remittances, get_remittance
from app.schemas.remittance import RemittanceListResponse, RemittanceResponse

router = APIRouter(prefix="/remittances", tags=["remittances"])


@router.get("", response_model=RemittanceListResponse)
async def list_remittances(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    remittances, total = await get_remittances(skip=skip, limit=limit)
    return RemittanceListResponse(
        remittances=remittances, total=total, skip=skip, limit=limit,
    )


@router.get("/{remittance_id}", response_model=RemittanceResponse)
async def get_remittance_detail(remittance_id: str):
    remit = await get_remittance(remittance_id)
    if not remit:
        raise HTTPException(status_code=404, detail="Remittance not found")
    return remit
