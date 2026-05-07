"""Repository for claim_lifecycle and lifecycle_attempts tables."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, func, update, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import ClaimLifecycle, LifecycleAttempt


async def create_lifecycle(session: AsyncSession, doc: dict, first_attempt: dict) -> int:
    """Create a new lifecycle with its first attempt. Returns lifecycle id."""
    lc = ClaimLifecycle(
        claim_id=doc["claim_id"],
        payer_name=doc.get("payer_name"),
        patient_name=doc.get("patient_name"),
        current_status=doc.get("current_status", "PENDING"),
        total_attempts=1,
        created_at=doc.get("created_at", datetime.utcnow()),
        updated_at=doc.get("updated_at", datetime.utcnow()),
    )
    session.add(lc)
    await session.flush()

    attempt = LifecycleAttempt(lifecycle_id=lc.id, **first_attempt)
    session.add(attempt)
    await session.flush()
    return lc.id


async def find_lifecycle(session: AsyncSession, claim_id: str) -> dict | None:
    """Find lifecycle by claim_id with all attempts."""
    result = await session.execute(
        select(ClaimLifecycle)
        .options(selectinload(ClaimLifecycle.attempts))
        .where(ClaimLifecycle.claim_id == claim_id)
    )
    lc = result.scalar_one_or_none()
    if not lc:
        return None
    return _to_dict(lc)


async def add_attempt(session: AsyncSession, claim_id: str, attempt_data: dict, update_fields: dict) -> None:
    """Add a new attempt to an existing lifecycle."""
    # Get lifecycle id
    result = await session.execute(
        select(ClaimLifecycle.id).where(ClaimLifecycle.claim_id == claim_id)
    )
    lc_id = result.scalar_one_or_none()
    if lc_id is None:
        return

    attempt = LifecycleAttempt(lifecycle_id=lc_id, **attempt_data)
    session.add(attempt)

    # Update lifecycle fields
    await session.execute(
        update(ClaimLifecycle)
        .where(ClaimLifecycle.claim_id == claim_id)
        .values(**update_fields)
    )
    await session.flush()


async def update_attempt_outcome(
    session: AsyncSession,
    lifecycle_id: int,
    *,
    status: str,
    denial_codes: list[str] | None = None,
    paid_amount: float = 0.0,
    billed_amount: float | None = None,
    remittance_date: datetime | None = None,
) -> int | None:
    """Update the latest PENDING attempt with outcome. Returns attempt_number or None."""
    # Find latest PENDING attempt
    result = await session.execute(
        select(LifecycleAttempt)
        .where(
            LifecycleAttempt.lifecycle_id == lifecycle_id,
            LifecycleAttempt.status == "PENDING",
        )
        .order_by(LifecycleAttempt.attempt_number.desc())
        .limit(1)
    )
    attempt = result.scalar_one_or_none()
    if not attempt:
        return None

    attempt.status = status
    attempt.denial_codes = denial_codes or []
    attempt.paid_amount = paid_amount
    if billed_amount is not None:
        attempt.billed_amount = billed_amount
    attempt.remittance_date = remittance_date or datetime.utcnow()
    await session.flush()
    return attempt.attempt_number


async def update_lifecycle_status(session: AsyncSession, claim_id: str, status: str) -> None:
    await session.execute(
        update(ClaimLifecycle)
        .where(ClaimLifecycle.claim_id == claim_id)
        .values(current_status=status, updated_at=datetime.utcnow())
    )
    await session.flush()


async def get_lifecycle_detail(session: AsyncSession, claim_id: str) -> dict | None:
    """Alias for find_lifecycle."""
    return await find_lifecycle(session, claim_id)


async def get_lifecycles_paginated(
    session: AsyncSession,
    *,
    skip: int = 0,
    limit: int = 50,
    status: str | None = None,
    min_attempts: int | None = None,
    payer_name: str | None = None,
) -> tuple[list[dict], int]:
    stmt = select(ClaimLifecycle).options(selectinload(ClaimLifecycle.attempts))
    count_stmt = select(func.count()).select_from(ClaimLifecycle)

    if status:
        stmt = stmt.where(ClaimLifecycle.current_status == status.upper())
        count_stmt = count_stmt.where(ClaimLifecycle.current_status == status.upper())
    if min_attempts is not None and min_attempts > 0:
        stmt = stmt.where(ClaimLifecycle.total_attempts >= min_attempts)
        count_stmt = count_stmt.where(ClaimLifecycle.total_attempts >= min_attempts)
    if payer_name:
        stmt = stmt.where(ClaimLifecycle.payer_name.ilike(f"%{payer_name}%"))
        count_stmt = count_stmt.where(ClaimLifecycle.payer_name.ilike(f"%{payer_name}%"))

    total = (await session.execute(count_stmt)).scalar() or 0
    stmt = stmt.order_by(ClaimLifecycle.updated_at.desc()).offset(skip).limit(limit)
    result = await session.execute(stmt)

    return [_to_dict(lc) for lc in result.scalars().all()], total


async def get_stats(session: AsyncSession) -> dict:
    """Compute aggregate lifecycle statistics."""
    total = (await session.execute(
        select(func.count()).select_from(ClaimLifecycle)
    )).scalar() or 0

    if total == 0:
        return {
            "total_claims_tracked": 0,
            "first_pass_payment_rate": 0.0,
            "avg_attempts_to_payment": 0.0,
            "resubmission_success_rate": 0.0,
            "resubmission_count": 0,
            "status_breakdown": {},
            "payer_breakdown": [],
        }

    # First-pass payment rate: lifecycles where attempt_number=1 has status PAID
    first_paid = (await session.execute(
        select(func.count()).select_from(LifecycleAttempt)
        .where(LifecycleAttempt.attempt_number == 1, LifecycleAttempt.status == "PAID")
    )).scalar() or 0

    first_resolved = (await session.execute(
        select(func.count()).select_from(LifecycleAttempt)
        .where(
            LifecycleAttempt.attempt_number == 1,
            LifecycleAttempt.status.in_(["PAID", "DENIED", "PARTIAL"]),
        )
    )).scalar() or 0

    first_pass_rate = round(first_paid / first_resolved, 4) if first_resolved else 0.0

    # Avg attempts to payment
    avg_result = (await session.execute(
        select(func.avg(ClaimLifecycle.total_attempts))
        .where(ClaimLifecycle.current_status == "PAID")
    )).scalar()
    avg_attempts = round(float(avg_result), 2) if avg_result else 0.0

    # Resubmission success rate
    resub_total = (await session.execute(
        select(func.count()).select_from(ClaimLifecycle)
        .where(ClaimLifecycle.total_attempts > 1)
    )).scalar() or 0

    resub_paid = (await session.execute(
        select(func.count()).select_from(ClaimLifecycle)
        .where(ClaimLifecycle.total_attempts > 1, ClaimLifecycle.current_status == "PAID")
    )).scalar() or 0

    resub_rate = round(resub_paid / resub_total, 4) if resub_total else 0.0

    # Status breakdown
    status_result = await session.execute(
        select(ClaimLifecycle.current_status, func.count().label("cnt"))
        .group_by(ClaimLifecycle.current_status)
    )
    status_breakdown = {row.current_status: row.cnt for row in status_result.all()}

    # Payer breakdown
    payer_result = await session.execute(
        select(
            ClaimLifecycle.payer_name,
            func.count().label("total"),
            func.count().filter(ClaimLifecycle.current_status == "PAID").label("paid"),
            func.count().filter(ClaimLifecycle.current_status == "DENIED").label("denied"),
            func.count().filter(ClaimLifecycle.current_status == "PENDING").label("pending"),
            func.avg(ClaimLifecycle.total_attempts).label("avg_attempts"),
        )
        .group_by(ClaimLifecycle.payer_name)
        .order_by(func.count().desc())
        .limit(20)
    )
    payer_breakdown = [
        {
            "payer_name": row.payer_name,
            "total": row.total,
            "paid": row.paid,
            "denied": row.denied,
            "pending": row.pending,
            "avg_attempts": round(float(row.avg_attempts), 2) if row.avg_attempts else 0.0,
            "payment_rate": round(row.paid / row.total, 4) if row.total else 0.0,
        }
        for row in payer_result.all()
    ]

    return {
        "total_claims_tracked": total,
        "first_pass_payment_rate": first_pass_rate,
        "avg_attempts_to_payment": avg_attempts,
        "resubmission_success_rate": resub_rate,
        "resubmission_count": resub_total,
        "status_breakdown": status_breakdown,
        "payer_breakdown": payer_breakdown,
    }


def _attempt_to_dict(a: LifecycleAttempt) -> dict:
    return {
        "attempt_number": a.attempt_number,
        "attempt_type": a.attempt_type,
        "status": a.status,
        "submitted_at": a.submitted_at.isoformat() if a.submitted_at else None,
        "frequency_code": a.frequency_code,
        "fix_applied": a.fix_applied,
        "features": a.features or {},
        "service_lines": a.service_lines or [],
        "validation_issues": a.validation_issues or [],
        "fixes_recommended": a.fixes_recommended or [],
        "prediction_risk_score": a.prediction_risk_score,
        "prediction_risk_level": a.prediction_risk_level,
        "model_version": a.model_version,
        "denial_codes": a.denial_codes or [],
        "paid_amount": a.paid_amount,
        "billed_amount": a.billed_amount,
        "remittance_date": a.remittance_date.isoformat() if a.remittance_date else None,
    }


def _to_dict(lc: ClaimLifecycle) -> dict:
    return {
        "id": lc.id,
        "claim_id": lc.claim_id,
        "payer_name": lc.payer_name,
        "patient_name": lc.patient_name,
        "current_status": lc.current_status,
        "total_attempts": lc.total_attempts,
        "created_at": lc.created_at.isoformat() if lc.created_at else None,
        "updated_at": lc.updated_at.isoformat() if lc.updated_at else None,
        "attempts": [_attempt_to_dict(a) for a in (lc.attempts or [])],
    }
