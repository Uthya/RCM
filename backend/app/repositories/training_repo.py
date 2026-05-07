"""Repository for ml_training_data, ml_training_data_archive, and training_history."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MLTrainingData, MLTrainingDataArchive, TrainingHistory


async def find_existing(session: AsyncSession, claim_id: str) -> bool:
    """Check if a training record exists for a claim."""
    result = await session.execute(
        select(MLTrainingData.id).where(MLTrainingData.claim_id == claim_id).limit(1)
    )
    return result.scalar_one_or_none() is not None


async def insert_record(session: AsyncSession, doc: dict) -> None:
    session.add(MLTrainingData(**doc))
    await session.flush()


async def get_training_data(
    session: AsyncSession,
    *,
    created_after: datetime | None = None,
    feature_version: str | None = None,
    limit: int = 100_000,
) -> list[dict]:
    """Fetch training records with optional filters."""
    stmt = select(MLTrainingData)
    if created_after:
        stmt = stmt.where(MLTrainingData.created_at >= created_after)
    if feature_version:
        stmt = stmt.where(MLTrainingData.feature_version == feature_version)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return [_to_dict(r) for r in result.scalars().all()]


async def count_records(session: AsyncSession, label: int | None = None) -> int:
    stmt = select(func.count()).select_from(MLTrainingData)
    if label is not None:
        stmt = stmt.where(MLTrainingData.label == label)
    result = await session.execute(stmt)
    return result.scalar() or 0


async def count_empty_features(session: AsyncSession) -> int:
    """Count records where features is NULL or empty."""
    result = await session.execute(
        select(func.count()).select_from(MLTrainingData)
        .where(
            (MLTrainingData.features.is_(None)) | (MLTrainingData.features == {})
        )
    )
    return result.scalar() or 0


async def count_archived(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(MLTrainingDataArchive))
    return result.scalar() or 0


async def archive_old(session: AsyncSession, cutoff: datetime) -> int:
    """Move records older than cutoff to archive. Returns count archived."""
    stmt = select(MLTrainingData).where(MLTrainingData.created_at < cutoff)
    result = await session.execute(stmt)
    old_records = result.scalars().all()

    if not old_records:
        return 0

    for r in old_records:
        session.add(MLTrainingDataArchive(
            claim_id=r.claim_id,
            attempt_number=r.attempt_number,
            is_first_attempt=r.is_first_attempt,
            features=r.features,
            label=r.label,
            actual_outcome=r.actual_outcome,
            denial_code=r.denial_code,
            denial_code_description=r.denial_code_description,
            all_carc_codes=r.all_carc_codes,
            paid_amount=r.paid_amount,
            billed_amount=r.billed_amount,
            model_version_at_prediction=r.model_version_at_prediction,
            prediction_risk_score=r.prediction_risk_score,
            feature_version=r.feature_version,
            feature_count=r.feature_count,
            feature_hash=r.feature_hash,
            created_at=r.created_at,
        ))

    count = len(old_records)
    await session.execute(delete(MLTrainingData).where(MLTrainingData.created_at < cutoff))
    await session.flush()
    return count


async def insert_history(session: AsyncSession, doc: dict) -> None:
    session.add(TrainingHistory(**doc))
    await session.flush()


async def get_latest_history(session: AsyncSession) -> dict | None:
    result = await session.execute(
        select(TrainingHistory).order_by(TrainingHistory.trained_at.desc()).limit(1)
    )
    h = result.scalar_one_or_none()
    if not h:
        return None
    return {
        "trained_at": h.trained_at,
        "model_version": h.model_version,
        "real_samples": h.real_samples,
        "synthetic_samples": h.synthetic_samples,
        "total_samples": h.total_samples,
        "metrics": h.metrics,
        "elapsed_seconds": h.elapsed_seconds,
    }


async def get_top_denial_codes(session: AsyncSession, limit: int = 10) -> list[dict]:
    """Get top denial codes from training data."""
    from sqlalchemy import literal_column
    stmt = (
        select(
            MLTrainingData.denial_code,
            func.count().label("count"),
        )
        .where(MLTrainingData.label == 1, MLTrainingData.denial_code.isnot(None))
        .group_by(MLTrainingData.denial_code)
        .order_by(func.count().desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [{"code": row.denial_code, "count": row.count} for row in result.all()]


async def get_version_distribution(session: AsyncSession) -> list[dict]:
    stmt = (
        select(
            MLTrainingData.model_version_at_prediction,
            func.count().label("count"),
        )
        .group_by(MLTrainingData.model_version_at_prediction)
        .order_by(func.count().desc())
    )
    result = await session.execute(stmt)
    return [{"version": row.model_version_at_prediction, "count": row.count} for row in result.all()]


async def upsert_training_record(session: AsyncSession, doc: dict) -> None:
    """Upsert for backfill — insert or update by claim_id."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    stmt = pg_insert(MLTrainingData).values(**doc)
    update_cols = {k: v for k, v in doc.items() if k not in ("claim_id", "attempt_number")}
    stmt = stmt.on_conflict_do_update(
        constraint="uq_ml_training_data_claim_attempt",
        set_=update_cols,
    )
    await session.execute(stmt)
    await session.flush()


async def get_all_claim_ids(session: AsyncSession) -> set[str]:
    result = await session.execute(select(MLTrainingData.claim_id))
    return {row[0] for row in result.all()}


def _to_dict(r: MLTrainingData) -> dict:
    return {
        "id": r.id,
        "claim_id": r.claim_id,
        "attempt_number": r.attempt_number,
        "is_first_attempt": r.is_first_attempt,
        "features": r.features or {},
        "label": r.label,
        "actual_outcome": r.actual_outcome,
        "denial_code": r.denial_code,
        "denial_code_description": r.denial_code_description,
        "all_carc_codes": r.all_carc_codes or [],
        "paid_amount": r.paid_amount,
        "billed_amount": r.billed_amount,
        "model_version_at_prediction": r.model_version_at_prediction,
        "prediction_risk_score": r.prediction_risk_score,
        "feature_version": r.feature_version,
        "feature_count": r.feature_count,
        "feature_hash": r.feature_hash,
        "created_at": r.created_at,
    }
