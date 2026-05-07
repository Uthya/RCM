"""Aggregate statistics for dashboard."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import stats_repo
from app.schemas.dashboard import (
    DashboardSummary, RiskDistribution, RiskBucket,
    PayerStatsResponse, PayerStat,
)


async def get_summary(session: AsyncSession) -> DashboardSummary:
    counts = await stats_repo.get_summary_counts(session)
    denial_rate = await stats_repo.get_denial_rate(session)
    total_billed = await stats_repo.get_total_billed(session)
    total_paid = await stats_repo.get_total_paid(session)

    return DashboardSummary(
        total_claims=counts["total_claims"],
        total_predicted=counts["total_predicted"],
        high_risk_count=counts["high_risk"],
        medium_risk_count=counts["medium_risk"],
        low_risk_count=counts["low_risk"],
        total_remittances=counts["total_remittances"],
        denial_rate=round(denial_rate, 4),
        total_billed=round(total_billed, 2),
        total_paid=round(total_paid, 2),
    )


async def get_risk_distribution(session: AsyncSession) -> RiskDistribution:
    buckets = await stats_repo.get_risk_distribution(session)
    return RiskDistribution(
        buckets=[RiskBucket(range_label=b["range_label"], count=b["count"]) for b in buckets]
    )


async def get_payer_stats(session: AsyncSession) -> PayerStatsResponse:
    raw = await stats_repo.get_payer_stats(session)
    payers = []
    for r in raw:
        total = r["total_claims"]
        denied = r["denied_count"]
        payers.append(PayerStat(
            payer_name=r["payer_name"],
            payer_id="",
            total_claims=total,
            denied_count=denied,
            denial_rate=round(denied / total, 4) if total > 0 else 0.0,
        ))
    return PayerStatsResponse(payers=payers)
