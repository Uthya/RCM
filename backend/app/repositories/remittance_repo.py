"""Repository for remittances table."""

from __future__ import annotations

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Remittance


async def insert_remittance(session: AsyncSession, doc: dict) -> int:
    """Insert a remittance and return the new id."""
    r = Remittance(**doc)
    session.add(r)
    await session.flush()
    return r.id


async def get_remittances_paginated(
    session: AsyncSession,
    *,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[dict], int]:
    total_result = await session.execute(select(func.count()).select_from(Remittance))
    total = total_result.scalar() or 0

    result = await session.execute(
        select(Remittance).order_by(Remittance.created_at.desc()).offset(skip).limit(limit)
    )
    rows = result.scalars().all()
    return [_to_dict(r) for r in rows], total


async def get_remittance_by_id(session: AsyncSession, remittance_id: int) -> dict | None:
    r = await session.get(Remittance, remittance_id)
    if not r:
        return None
    return _to_dict(r)


async def find_remittance_by_claim_id(session: AsyncSession, claim_id: str) -> dict | None:
    result = await session.execute(
        select(Remittance).where(Remittance.claim_id == claim_id).order_by(Remittance.created_at.desc())
    )
    r = result.scalars().first()
    if not r:
        return None
    return _to_dict(r)


async def count_remittances(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Remittance))
    return result.scalar() or 0


async def count_denied(session: AsyncSession) -> int:
    result = await session.execute(
        select(func.count()).select_from(Remittance).where(Remittance.claim_status == "denied")
    )
    return result.scalar() or 0


DENIAL_PRIOR = 0.15
DENIAL_PRIOR_STRENGTH = 20


async def get_denial_rate(session: AsyncSession, filter_col: str, filter_val: str) -> float:
    """Get denial rate for a given filter (e.g., payer_name, payee_npi)."""
    col = getattr(Remittance, filter_col, None)
    if col is None:
        return 0.0
    total_result = await session.execute(
        select(func.count()).select_from(Remittance).where(col == filter_val)
    )
    total = total_result.scalar() or 0
    if total == 0:
        return 0.0
    denied_result = await session.execute(
        select(func.count()).select_from(Remittance)
        .where(col == filter_val, Remittance.claim_status == "denied")
    )
    denied = denied_result.scalar() or 0
    return denied / total


async def get_denial_rate_with_count(
    session: AsyncSession, filter_col: str, filter_val: str,
) -> tuple[float, int]:
    """Bayesian-smoothed denial rate + sample count."""
    col = getattr(Remittance, filter_col, None)
    if col is None:
        return DENIAL_PRIOR, 0
    total_result = await session.execute(
        select(func.count()).select_from(Remittance).where(col == filter_val)
    )
    total = total_result.scalar() or 0
    if total == 0:
        return DENIAL_PRIOR, 0
    denied_result = await session.execute(
        select(func.count()).select_from(Remittance)
        .where(col == filter_val, Remittance.claim_status == "denied")
    )
    denied = denied_result.scalar() or 0
    smoothed = (denied + DENIAL_PRIOR * DENIAL_PRIOR_STRENGTH) / (total + DENIAL_PRIOR_STRENGTH)
    return smoothed, total


async def get_cpt_denial_rate(session: AsyncSession, cpt_code: str) -> float:
    """Get denial rate for a specific CPT code using JSONB service_lines."""
    stmt = text("""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE r.claim_status = 'denied') AS denied
        FROM remittances r, jsonb_array_elements(r.service_lines) AS sl
        WHERE sl->>'cpt_code' = :cpt_code
    """)
    result = await session.execute(stmt, {"cpt_code": cpt_code})
    row = result.one()
    if row.total == 0:
        return 0.0
    return row.denied / row.total


async def get_cpt_denial_rate_with_count(
    session: AsyncSession, cpt_code: str,
) -> tuple[float, int]:
    """Bayesian-smoothed CPT denial rate + sample count."""
    stmt = text("""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE r.claim_status = 'denied') AS denied
        FROM remittances r, jsonb_array_elements(r.service_lines) AS sl
        WHERE sl->>'cpt_code' = :cpt_code
    """)
    result = await session.execute(stmt, {"cpt_code": cpt_code})
    row = result.one()
    if row.total == 0:
        return DENIAL_PRIOR, 0
    smoothed = (row.denied + DENIAL_PRIOR * DENIAL_PRIOR_STRENGTH) / (row.total + DENIAL_PRIOR_STRENGTH)
    return smoothed, row.total


async def get_total_paid(session: AsyncSession) -> float:
    result = await session.execute(
        select(func.coalesce(func.sum(Remittance.paid_amount), 0.0))
    )
    return float(result.scalar() or 0.0)


async def get_payer_stats(session: AsyncSession) -> list[dict]:
    """Get denial counts grouped by payer_name."""
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


def _to_dict(r: Remittance) -> dict:
    return {
        "id": r.id,
        "claim_id": r.claim_id,
        "payer_control_number": r.payer_control_number,
        "claim_status_code": r.claim_status_code,
        "claim_status": r.claim_status,
        "billed_amount": r.billed_amount,
        "paid_amount": r.paid_amount,
        "patient_responsibility": r.patient_responsibility,
        "payer_name": r.payer_name,
        "payee_name": r.payee_name,
        "payee_npi": r.payee_npi,
        "total_payment_amount": r.total_payment_amount,
        "payment_method": r.payment_method,
        "payment_date": r.payment_date,
        "trace_number": r.trace_number,
        "adjustments": r.adjustments or [],
        "carc_codes": r.carc_codes or [],
        "rarc_codes": r.rarc_codes or [],
        "service_lines": r.service_lines or [],
        "created_at": r.created_at,
    }
