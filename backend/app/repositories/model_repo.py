"""Repository for model_registry table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ModelRegistry


async def get_latest_version(session: AsyncSession) -> int | None:
    result = await session.execute(
        select(ModelRegistry.version).order_by(ModelRegistry.version.desc()).limit(1)
    )
    return result.scalar_one_or_none()


async def upsert_model(session: AsyncSession, doc: dict) -> None:
    stmt = pg_insert(ModelRegistry).values(**doc)
    update_cols = {k: v for k, v in doc.items() if k != "version"}
    stmt = stmt.on_conflict_do_update(
        index_elements=["version"],
        set_=update_cols,
    )
    await session.execute(stmt)
    await session.flush()


async def get_active(session: AsyncSession) -> dict | None:
    result = await session.execute(
        select(ModelRegistry).where(ModelRegistry.is_active == True)
    )
    m = result.scalar_one_or_none()
    if not m:
        return None
    return _to_dict(m)


async def find_by_version(session: AsyncSession, version: int) -> dict | None:
    result = await session.execute(
        select(ModelRegistry).where(ModelRegistry.version == version)
    )
    m = result.scalar_one_or_none()
    if not m:
        return None
    return _to_dict(m)


async def deactivate_all(session: AsyncSession) -> None:
    await session.execute(
        update(ModelRegistry)
        .where(ModelRegistry.is_active == True)
        .values(is_active=False, status="retired")
    )
    await session.flush()


async def activate(session: AsyncSession, version: int) -> None:
    await session.execute(
        update(ModelRegistry)
        .where(ModelRegistry.version == version)
        .values(is_active=True, status="active", promoted_at=datetime.utcnow())
    )
    await session.flush()


async def set_status(session: AsyncSession, version: int, status: str) -> None:
    await session.execute(
        update(ModelRegistry)
        .where(ModelRegistry.version == version)
        .values(status=status, is_active=False)
    )
    await session.flush()


async def get_previous_retired(session: AsyncSession) -> dict | None:
    result = await session.execute(
        select(ModelRegistry)
        .where(ModelRegistry.status == "retired")
        .order_by(ModelRegistry.version.desc())
        .limit(1)
    )
    m = result.scalar_one_or_none()
    if not m:
        return None
    return _to_dict(m)


async def list_versions(session: AsyncSession, limit: int = 100) -> list[dict]:
    result = await session.execute(
        select(ModelRegistry).order_by(ModelRegistry.version.desc()).limit(limit)
    )
    return [_to_dict(m) for m in result.scalars().all()]


def _to_dict(m: ModelRegistry) -> dict:
    return {
        "id": m.id,
        "version": m.version,
        "version_str": m.version_str or f"v{m.version}",
        "is_active": m.is_active,
        "status": m.status,
        "model_path": m.model_path,
        "trained_at": m.trained_at,
        "promoted_at": m.promoted_at,
        "real_samples": m.real_samples,
        "synthetic_samples": m.synthetic_samples,
        "feature_version": m.feature_version,
        "feature_count": m.feature_count,
        "feature_hash": m.feature_hash,
        "metrics": m.metrics or {},
        "feature_importance": m.feature_importance or {},
        "top_denial_codes": m.top_denial_codes or [],
    }
