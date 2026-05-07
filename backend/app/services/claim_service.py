"""CRUD operations for parsed 837 claims."""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import claim_repo, prediction_repo, outcome_repo, remittance_repo
from app.schemas.claim import ParsedClaim, ClaimResponse


async def store_claims(session: AsyncSession, claims: list[ParsedClaim]) -> int:
    """Insert parsed 837 claims. Returns count inserted."""
    inserted = 0
    for claim in claims:
        doc = claim.model_dump()
        doc["created_at"] = datetime.utcnow()
        await claim_repo.upsert_claim(session, doc)
        inserted += 1
    return inserted


async def get_claims(
    session: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    risk_level: str | None = None,
    payer_id: str | None = None,
    sort_by: str = "created_at",
    sort_order: int = -1,
) -> tuple[list[ClaimResponse], int]:
    """Get paginated claims list with optional filters. Joins prediction data."""
    rows, total = await claim_repo.get_claims_joined(
        session,
        skip=skip, limit=limit,
        risk_level=risk_level, payer_id=payer_id,
        sort_by=sort_by, sort_order=sort_order,
    )

    claims = []
    for row in rows:
        pred = row.get("prediction")
        outcome = row.get("outcome")
        claims.append(ClaimResponse(
            claim_id=row.get("claim_id", ""),
            patient_name=f"{row.get('patient_last_name', '')}, {row.get('patient_first_name', '')}".strip(", "),
            payer_name=row.get("payer_name", ""),
            payer_id=row.get("payer_id", ""),
            total_charge=row.get("total_charge", 0.0),
            diagnosis_codes=row.get("diagnosis_codes", []),
            service_lines=row.get("service_lines", []),
            place_of_service=row.get("place_of_service", ""),
            billing_provider_name=row.get("billing_provider_name", ""),
            billing_provider_npi=row.get("billing_provider_npi", ""),
            patient_dob=row.get("patient_dob", ""),
            patient_gender=row.get("patient_gender", ""),
            provider_taxonomy=row.get("provider_taxonomy", ""),
            prior_auth_number=row.get("prior_auth_number", ""),
            created_at=row.get("created_at"),
            risk_score=pred.get("risk_score") if pred else None,
            risk_level=pred.get("risk_level") if pred else None,
            risk_factors=pred.get("risk_factors") if pred else None,
            actual_outcome=outcome.get("outcome_status") if outcome else None,
            paid_amount=outcome.get("paid_amount") if outcome else None,
            carc_codes=outcome.get("carc_codes") if outcome else None,
        ))

    return claims, total


async def get_claim(session: AsyncSession, claim_id: str) -> ClaimResponse | None:
    """Get single claim with prediction, outcome, and remittance data."""
    doc = await claim_repo.get_claim_detail(session, claim_id)
    if not doc:
        return None

    pred = await prediction_repo.find_prediction(session, claim_id)
    outcome = await outcome_repo.find_outcome(session, claim_id)
    remit = await remittance_repo.find_remittance_by_claim_id(session, claim_id) if not outcome else None

    actual_outcome = None
    paid_amount = None
    carc_codes = None
    if outcome:
        actual_outcome = outcome.get("outcome_status")
        paid_amount = outcome.get("paid_amount")
        carc_codes = outcome.get("carc_codes")
    elif remit:
        actual_outcome = remit.get("claim_status")
        paid_amount = remit.get("paid_amount")
        carc_codes = remit.get("carc_codes")

    return ClaimResponse(
        claim_id=doc.get("claim_id", ""),
        patient_name=f"{doc.get('patient_last_name', '')}, {doc.get('patient_first_name', '')}".strip(", "),
        payer_name=doc.get("payer_name", ""),
        payer_id=doc.get("payer_id", ""),
        total_charge=doc.get("total_charge", 0.0),
        diagnosis_codes=doc.get("diagnosis_codes", []),
        service_lines=doc.get("service_lines", []),
        place_of_service=doc.get("place_of_service", ""),
        billing_provider_name=doc.get("billing_provider_name", ""),
        billing_provider_npi=doc.get("billing_provider_npi", ""),
        patient_dob=doc.get("patient_dob", ""),
        patient_gender=doc.get("patient_gender", ""),
        provider_taxonomy=doc.get("provider_taxonomy", ""),
        prior_auth_number=doc.get("prior_auth_number", ""),
        created_at=doc.get("created_at"),
        risk_score=pred.get("risk_score") if pred else None,
        risk_level=pred.get("risk_level") if pred else None,
        risk_factors=pred.get("risk_factors") if pred else None,
        actual_outcome=actual_outcome,
        paid_amount=paid_amount,
        carc_codes=carc_codes,
    )
