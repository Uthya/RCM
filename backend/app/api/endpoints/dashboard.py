from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.services.stats_service import get_summary, get_risk_distribution, get_payer_stats
from app.schemas.dashboard import DashboardSummary, RiskDistribution, PayerStatsResponse

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary(session: AsyncSession = Depends(get_db)):
    return await get_summary(session)


@router.get("/risk-distribution", response_model=RiskDistribution)
async def risk_distribution(session: AsyncSession = Depends(get_db)):
    return await get_risk_distribution(session)


@router.get("/payer-stats", response_model=PayerStatsResponse)
async def payer_stats(session: AsyncSession = Depends(get_db)):
    return await get_payer_stats(session)
