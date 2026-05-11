"""Repository for claim_outcomes table."""

from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ClaimOutcome


async def upsert_outcome(session: AsyncSession, doc: dict) -> bool:
    """Upsert a claim outcome. Returns True if it was a new insert."""
    # Check if exists
    result = await session.execute(
        select(ClaimOutcome.id).where(
            ClaimOutcome.claim_id == doc["claim_id"],
            ClaimOutcome.attempt_number == doc.get("attempt_number", 1),
        )
    )
    is_new = result.scalar_one_or_none() is None

    stmt = pg_insert(ClaimOutcome).values(**doc)
    update_cols = {k: v for k, v in doc.items() if k not in ("claim_id", "attempt_number")}
    stmt = stmt.on_conflict_do_update(
        constraint="uq_claim_outcomes_claim_attempt",
        set_=update_cols,
    )
    await session.execute(stmt)
    await session.flush()
    return is_new


async def find_outcome(
    session: AsyncSession, claim_id: str, attempt_number: int | None = None,
) -> dict | None:
    stmt = select(ClaimOutcome).where(ClaimOutcome.claim_id == claim_id)
    if attempt_number is not None:
        stmt = stmt.where(ClaimOutcome.attempt_number == attempt_number)
    stmt = stmt.order_by(ClaimOutcome.attempt_number.desc())
    result = await session.execute(stmt)
    o = result.scalars().first()
    if not o:
        return None
    return _to_dict(o)


async def count_outcomes(session: AsyncSession, status_filter: list[str] | None = None) -> int:
    stmt = select(func.count()).select_from(ClaimOutcome)
    if status_filter:
        stmt = stmt.where(ClaimOutcome.outcome_status.in_(status_filter))
    result = await session.execute(stmt)
    return result.scalar() or 0


def _to_dict(o: ClaimOutcome) -> dict:
    return {
        "id": o.id,
        "claim_id": o.claim_id,
        "attempt_number": o.attempt_number,
        "outcome_status": o.outcome_status,
        "paid_amount": o.paid_amount,
        "carc_codes": o.carc_codes or [],
        "carc_descriptions": o.carc_descriptions or [],
        "model_version": o.model_version,
        "created_at": o.created_at,
    }
