"""Repository for decision_config, cpt_risk_config, and upload_history."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DecisionConfig, CptRiskConfig, UploadHistory


async def get_payer_weights(session: AsyncSession) -> dict | None:
    """Get payer weights from decision_config."""
    result = await session.execute(
        select(DecisionConfig).where(DecisionConfig.type == "payer_weights")
    )
    config = result.scalar_one_or_none()
    if not config:
        return None
    return config.weights


async def get_cpt_patterns(session: AsyncSession) -> list[dict]:
    """Get CPT risk patterns."""
    result = await session.execute(select(CptRiskConfig))
    rows = result.scalars().all()
    return [
        {
            "cpt_prefix": r.cpt_prefix,
            "weight": r.weight,
            "label": r.label,
            "reason": r.reason,
        }
        for r in rows
    ]


async def insert_upload_record(session: AsyncSession, doc: dict) -> None:
    session.add(UploadHistory(**doc))
    await session.flush()
