from fastapi import APIRouter

from app.services.stats_service import get_summary, get_risk_distribution, get_payer_stats
from app.schemas.dashboard import DashboardSummary, RiskDistribution, PayerStatsResponse

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary():
    return await get_summary()


@router.get("/risk-distribution", response_model=RiskDistribution)
async def risk_distribution():
    return await get_risk_distribution()


@router.get("/payer-stats", response_model=PayerStatsResponse)
async def payer_stats():
    return await get_payer_stats()
