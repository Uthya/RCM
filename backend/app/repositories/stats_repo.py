"""Repository for cross-table dashboard statistics."""

from __future__ import annotations

from sqlalchemy import select, func, case, Float, cast
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Claim, Prediction, Remittance


async def get_summary_counts(session: AsyncSession) -> dict:
    total_claims = (await session.execute(
        select(func.count()).select_from(Claim)
    )).scalar() or 0

    total_predicted = (await session.execute(
        select(func.count()).select_from(Prediction)
    )).scalar() or 0

    total_remittances = (await session.execute(
        select(func.count()).select_from(Remittance)
    )).scalar() or 0

    high_risk = (await session.execute(
        select(func.count()).select_from(Prediction).where(Prediction.risk_level == "HIGH")
    )).scalar() or 0

    medium_risk = (await session.execute(
        select(func.count()).select_from(Prediction).where(Prediction.risk_level == "MEDIUM")
    )).scalar() or 0

    low_risk = (await session.execute(
        select(func.count()).select_from(Prediction).where(Prediction.risk_level == "LOW")
    )).scalar() or 0

    return {
        "total_claims": total_claims,
        "total_predicted": total_predicted,
        "total_remittances": total_remittances,
        "high_risk": high_risk,
        "medium_risk": medium_risk,
        "low_risk": low_risk,
    }


async def get_denial_rate(session: AsyncSession) -> float:
    total = (await session.execute(
        select(func.count()).select_from(Remittance)
    )).scalar() or 0
    if total == 0:
        return 0.0
    denied = (await session.execute(
        select(func.count()).select_from(Remittance).where(Remittance.claim_status == "denied")
    )).scalar() or 0
    return denied / total


async def get_risk_distribution(session: AsyncSession) -> list[dict]:
    """Risk score histogram — 10 buckets from 0.0-1.0."""
    bucket = case(
        (Prediction.risk_score < 0.1, "0.0-0.1"),
        (Prediction.risk_score < 0.2, "0.1-0.2"),
        (Prediction.risk_score < 0.3, "0.2-0.3"),
        (Prediction.risk_score < 0.4, "0.3-0.4"),
        (Prediction.risk_score < 0.5, "0.4-0.5"),
        (Prediction.risk_score < 0.6, "0.5-0.6"),
        (Prediction.risk_score < 0.7, "0.6-0.7"),
        (Prediction.risk_score < 0.8, "0.7-0.8"),
        (Prediction.risk_score < 0.9, "0.8-0.9"),
        else_="0.9-1.0",
    ).label("range_label")

    stmt = (
        select(bucket, func.count().label("count"))
        .select_from(Prediction)
        .group_by(bucket)
    )
    result = await session.execute(stmt)
    bucket_counts = {row.range_label: row.count for row in result.all()}

    # Ensure all buckets present
    boundaries = [
        "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
        "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0",
    ]
    return [{"range_label": b, "count": bucket_counts.get(b, 0)} for b in boundaries]


async def get_total_billed(session: AsyncSession) -> float:
    result = await session.execute(
        select(func.coalesce(func.sum(Claim.total_charge), 0.0))
    )
    return float(result.scalar() or 0.0)


async def get_total_paid(session: AsyncSession) -> float:
    result = await session.execute(
        select(func.coalesce(func.sum(Remittance.paid_amount), 0.0))
    )
    return float(result.scalar() or 0.0)


async def get_payer_stats(session: AsyncSession) -> list[dict]:
    stmt = (
        select(
            Remittance.payer_name,
            func.count().label("total_claims"),
            func.count().filter(Remittance.claim_status == "denied").label("denied_count"),
        )
        .group_by(Remittance.payer_name)
        .order_by(func.count().desc())
    )
    result = await session.execute(stmt)
    return [
        {
            "payer_name": row.payer_name or "Unknown",
            "total_claims": row.total_claims,
            "denied_count": row.denied_count,
        }
        for row in result.all()
    ]
