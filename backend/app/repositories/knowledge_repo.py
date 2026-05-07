"""Repository for fix_history and fix_effectiveness tables."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, func, Float, cast, case
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FixHistory, FixEffectiveness


async def insert_fix(session: AsyncSession, doc: dict) -> None:
    """Insert a raw fix history record."""
    session.add(FixHistory(**doc))
    await session.flush()


async def upsert_effectiveness(session: AsyncSession, doc: dict) -> None:
    """Atomically upsert fix effectiveness with counter increments."""
    is_success = 1 if doc["outcome"] == "paid" else 0
    is_failure = 1 - is_success

    base = {
        "payer_name": doc["payer_name"],
        "cpt_code": doc.get("cpt_code", ""),
        "issue_type": doc["issue_type"],
        "fix_applied": doc["fix_applied"],
        "total": 1,
        "success": is_success,
        "failure": is_failure,
        "success_rate": float(is_success),
        "confidence_level": "LOW",
        "updated_at": datetime.utcnow(),
    }

    stmt = pg_insert(FixEffectiveness).values(**base)
    new_total = FixEffectiveness.total + 1
    new_success = FixEffectiveness.success + is_success
    new_failure = FixEffectiveness.failure + is_failure

    stmt = stmt.on_conflict_do_update(
        constraint="uq_fix_effectiveness_composite",
        set_={
            "total": new_total,
            "success": new_success,
            "failure": new_failure,
            "success_rate": cast(new_success, Float) / cast(new_total, Float),
            "confidence_level": case(
                (new_total >= 50, "HIGH"),
                (new_total >= 10, "MEDIUM"),
                else_="LOW",
            ),
            "updated_at": datetime.utcnow(),
        },
    )
    await session.execute(stmt)
    await session.flush()


async def get_best_fix(
    session: AsyncSession,
    issue_type: str,
    payer_name: str,
    cpt_code: str = "",
    min_samples: int = 10,
) -> dict | None:
    stmt = (
        select(FixEffectiveness)
        .where(
            FixEffectiveness.issue_type == issue_type,
            FixEffectiveness.payer_name == payer_name,
            FixEffectiveness.total >= min_samples,
            FixEffectiveness.success_rate > 0.5,
        )
    )
    if cpt_code:
        stmt = stmt.where(FixEffectiveness.cpt_code == cpt_code)

    stmt = stmt.order_by(FixEffectiveness.success_rate.desc()).limit(1)
    result = await session.execute(stmt)
    r = result.scalar_one_or_none()
    if not r:
        return None

    return {
        "fix": r.fix_applied,
        "success_rate": round(r.success_rate, 2),
        "sample_size": r.total,
        "confidence": "high" if r.total >= 50 else "moderate",
    }


async def get_top_fixes(
    session: AsyncSession,
    issue_type: str,
    payer_name: str,
    cpt_code: str = "",
    limit: int = 3,
    min_samples: int = 10,
) -> list[dict]:
    stmt = (
        select(FixEffectiveness)
        .where(
            FixEffectiveness.issue_type == issue_type,
            FixEffectiveness.payer_name == payer_name,
            FixEffectiveness.total >= min_samples,
            FixEffectiveness.success_rate > 0.5,
        )
    )
    if cpt_code:
        stmt = stmt.where(FixEffectiveness.cpt_code == cpt_code)

    stmt = stmt.order_by(FixEffectiveness.success_rate.desc()).limit(limit)
    result = await session.execute(stmt)
    return [
        {
            "fix": r.fix_applied,
            "success_rate": round(r.success_rate, 2),
            "sample_size": r.total,
            "confidence": "high" if r.total >= 50 else "moderate",
        }
        for r in result.scalars().all()
    ]


async def get_fix_stats(
    session: AsyncSession,
    payer_name: str | None = None,
    issue_type: str | None = None,
    min_samples: int = 10,
) -> dict:
    stmt = select(FixEffectiveness)
    if payer_name:
        stmt = stmt.where(FixEffectiveness.payer_name == payer_name)
    if issue_type:
        stmt = stmt.where(FixEffectiveness.issue_type == issue_type)
    stmt = stmt.order_by(FixEffectiveness.success_rate.desc()).limit(500)

    result = await session.execute(stmt)
    all_results = result.scalars().all()

    qualified = []
    learning = []
    total_fix_records = 0
    for r in all_results:
        total_fix_records += r.total
        entry = {
            "payer_name": r.payer_name,
            "cpt_code": r.cpt_code,
            "issue_type": r.issue_type,
            "fix_applied": r.fix_applied,
            "success_rate": round(r.success_rate, 2),
            "sample_size": r.total,
        }
        if r.total >= min_samples:
            qualified.append(entry)
        else:
            learning.append(entry)

    return {
        "qualified": qualified,
        "learning": learning,
        "total_fix_records": total_fix_records,
    }
