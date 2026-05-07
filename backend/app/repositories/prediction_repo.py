"""Repository for predictions and shadow_predictions tables."""

from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Prediction, ShadowPrediction


async def upsert_prediction(session: AsyncSession, doc: dict) -> None:
    stmt = pg_insert(Prediction).values(**doc)
    update_cols = {k: v for k, v in doc.items() if k not in ("claim_id", "attempt_number")}
    stmt = stmt.on_conflict_do_update(
        constraint="uq_predictions_claim_attempt",
        set_=update_cols,
    )
    await session.execute(stmt)
    await session.flush()


async def find_prediction(session: AsyncSession, claim_id: str) -> dict | None:
    result = await session.execute(
        select(Prediction).where(Prediction.claim_id == claim_id).order_by(Prediction.attempt_number.desc())
    )
    p = result.scalar_one_or_none()
    if not p:
        return None
    return _to_dict(p)


async def get_predicted_ids(session: AsyncSession) -> list[str]:
    result = await session.execute(select(Prediction.claim_id).distinct())
    return [row[0] for row in result.all()]


async def get_unpredicted_claim_ids(session: AsyncSession) -> list[str]:
    """Return claim_ids that have no prediction."""
    from app.db.models import Claim
    sub = select(Prediction.claim_id).distinct()
    result = await session.execute(
        select(Claim.claim_id).where(Claim.claim_id.notin_(sub))
    )
    return [row[0] for row in result.all()]


async def update_prediction_fields(session: AsyncSession, claim_id: str, fields: dict) -> None:
    from sqlalchemy import update
    stmt = update(Prediction).where(Prediction.claim_id == claim_id).values(**fields)
    await session.execute(stmt)
    await session.flush()


async def upsert_shadow(session: AsyncSession, doc: dict) -> None:
    stmt = pg_insert(ShadowPrediction).values(**doc)
    update_cols = {k: v for k, v in doc.items() if k not in ("claim_id", "attempt_number")}
    stmt = stmt.on_conflict_do_update(
        constraint="uq_shadow_predictions_claim_attempt",
        set_=update_cols,
    )
    await session.execute(stmt)
    await session.flush()


async def get_shadows(session: AsyncSession, limit: int = 100_000) -> list[dict]:
    result = await session.execute(
        select(ShadowPrediction).limit(limit)
    )
    return [
        {
            "claim_id": s.claim_id,
            "active_score": s.active_score,
            "shadow_score": s.shadow_score,
            "active_version": s.active_version,
            "shadow_version": s.shadow_version,
            "actual_outcome": s.actual_outcome,
            "scored_at": s.scored_at,
        }
        for s in result.scalars().all()
    ]


async def get_shadows_with_outcomes(session: AsyncSession, limit: int = 100_000) -> list[dict]:
    result = await session.execute(
        select(ShadowPrediction)
        .where(ShadowPrediction.actual_outcome.isnot(None))
        .limit(limit)
    )
    return [
        {
            "claim_id": s.claim_id,
            "active_score": s.active_score,
            "shadow_score": s.shadow_score,
            "actual_outcome": s.actual_outcome,
        }
        for s in result.scalars().all()
    ]


def _to_dict(p: Prediction) -> dict:
    return {
        "claim_id": p.claim_id,
        "attempt_number": p.attempt_number,
        "risk_score": p.risk_score,
        "risk_level": p.risk_level,
        "features": p.features or {},
        "risk_factors": p.risk_factors or [],
        "feature_version": p.feature_version,
        "model_version": p.model_version,
        "action": p.action,
        "action_label": p.action_label,
        "created_at": p.created_at,
    }
