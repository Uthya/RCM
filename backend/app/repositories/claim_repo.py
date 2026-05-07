"""Repository for claims table."""

from __future__ import annotations

from sqlalchemy import select, func, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Claim, Prediction, ClaimOutcome


async def upsert_claim(session: AsyncSession, doc: dict) -> None:
    stmt = pg_insert(Claim).values(**doc)
    stmt = stmt.on_conflict_do_update(
        index_elements=["claim_id"],
        set_={k: v for k, v in doc.items() if k != "claim_id"},
    )
    await session.execute(stmt)
    await session.flush()


async def get_claims_joined(
    session: AsyncSession,
    *,
    skip: int = 0,
    limit: int = 50,
    risk_level: str | None = None,
    payer_id: str | None = None,
    sort_by: str = "created_at",
    sort_order: int = -1,
) -> tuple[list[dict], int]:
    """Return paginated claims joined with predictions and outcomes."""
    # Base query
    stmt = (
        select(Claim, Prediction, ClaimOutcome)
        .outerjoin(Prediction, Claim.claim_id == Prediction.claim_id)
        .outerjoin(ClaimOutcome, Claim.claim_id == ClaimOutcome.claim_id)
    )

    # Count query (same joins & filters, before pagination)
    count_stmt = (
        select(func.count())
        .select_from(Claim)
        .outerjoin(Prediction, Claim.claim_id == Prediction.claim_id)
        .outerjoin(ClaimOutcome, Claim.claim_id == ClaimOutcome.claim_id)
    )

    if payer_id:
        stmt = stmt.where(Claim.payer_id == payer_id)
        count_stmt = count_stmt.where(Claim.payer_id == payer_id)

    if risk_level:
        stmt = stmt.where(Prediction.risk_level == risk_level)
        count_stmt = count_stmt.where(Prediction.risk_level == risk_level)

    total = (await session.execute(count_stmt)).scalar() or 0

    # Sort
    if sort_by == "risk_score":
        sort_col = Prediction.risk_score
    else:
        sort_col = getattr(Claim, sort_by, Claim.created_at)

    if sort_order == -1:
        stmt = stmt.order_by(sort_col.desc().nullslast())
    else:
        stmt = stmt.order_by(sort_col.asc().nullsfirst())

    stmt = stmt.offset(skip).limit(limit)
    rows = (await session.execute(stmt)).all()

    results = []
    for claim, pred, outcome in rows:
        d = _claim_to_dict(claim)
        d["prediction"] = _prediction_to_dict(pred) if pred else None
        d["outcome"] = _outcome_to_dict(outcome) if outcome else None
        results.append(d)

    return results, total


async def get_claim_detail(session: AsyncSession, claim_id: str) -> dict | None:
    """Get single claim by claim_id."""
    result = await session.execute(
        select(Claim).where(Claim.claim_id == claim_id)
    )
    claim = result.scalar_one_or_none()
    if not claim:
        return None
    return _claim_to_dict(claim)


async def find_claim(session: AsyncSession, claim_id: str) -> dict | None:
    """Find a claim and return as dict."""
    return await get_claim_detail(session, claim_id)


async def get_claims_by_ids(session: AsyncSession, claim_ids: list[str]) -> list[dict]:
    """Get multiple claims by claim_ids."""
    result = await session.execute(
        select(Claim).where(Claim.claim_id.in_(claim_ids))
    )
    return [_claim_to_dict(c) for c in result.scalars().all()]


async def update_claim_fields(session: AsyncSession, claim_id: str, fields: dict) -> None:
    """Update specific fields on a claim."""
    stmt = update(Claim).where(Claim.claim_id == claim_id).values(**fields)
    await session.execute(stmt)
    await session.flush()


async def count_claims(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Claim))
    return result.scalar() or 0


def _claim_to_dict(c: Claim) -> dict:
    return {
        "id": c.id,
        "claim_id": c.claim_id,
        "sender_id": c.sender_id,
        "receiver_id": c.receiver_id,
        "transaction_date": c.transaction_date,
        "billing_provider_name": c.billing_provider_name,
        "billing_provider_npi": c.billing_provider_npi,
        "rendering_provider_name": c.rendering_provider_name,
        "rendering_provider_npi": c.rendering_provider_npi,
        "patient_first_name": c.patient_first_name,
        "patient_last_name": c.patient_last_name,
        "patient_dob": c.patient_dob,
        "patient_gender": c.patient_gender,
        "subscriber_id": c.subscriber_id,
        "payer_name": c.payer_name,
        "payer_id": c.payer_id,
        "payer_sequence": c.payer_sequence,
        "group_number": c.group_number,
        "total_charge": c.total_charge,
        "place_of_service": c.place_of_service,
        "frequency_code": c.frequency_code,
        "prior_auth_number": c.prior_auth_number,
        "provider_taxonomy": c.provider_taxonomy,
        "diagnosis_codes": c.diagnosis_codes or [],
        "service_lines": c.service_lines or [],
        "validation_issues": c.validation_issues or [],
        "issue_count": c.issue_count,
        "action": c.action,
        "action_label": c.action_label,
        "created_at": c.created_at,
    }


def _prediction_to_dict(p: Prediction) -> dict:
    return {
        "claim_id": p.claim_id,
        "risk_score": p.risk_score,
        "risk_level": p.risk_level,
        "risk_factors": p.risk_factors,
        "features": p.features,
        "model_version": p.model_version,
        "action": p.action,
        "action_label": p.action_label,
    }


def _outcome_to_dict(o: ClaimOutcome) -> dict:
    return {
        "claim_id": o.claim_id,
        "outcome_status": o.outcome_status,
        "paid_amount": o.paid_amount,
        "carc_codes": o.carc_codes or [],
        "carc_descriptions": o.carc_descriptions or [],
    }
