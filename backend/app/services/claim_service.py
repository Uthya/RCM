"""CRUD operations for parsed 837 claims in MongoDB."""

from datetime import datetime

from app.db.mongodb import get_db
from app.schemas.claim import ParsedClaim, ClaimResponse


async def store_claims(claims: list[ParsedClaim]) -> int:
    """Insert parsed 837 claims into MongoDB. Returns count inserted.
    Preserves outcome fields (actual_outcome, paid_amount, carc_codes)
    if the claim was already matched from an 835.
    """
    db = get_db()
    inserted = 0
    # Fields set by 835 matching — never overwrite these on re-upload
    PRESERVE_FIELDS = {"actual_outcome", "paid_amount", "carc_codes"}

    for claim in claims:
        doc = claim.model_dump()
        doc["created_at"] = datetime.utcnow()

        # Check if claim already exists with outcome data
        existing = await db.claims.find_one(
            {"claim_id": claim.claim_id},
            {f: 1 for f in PRESERVE_FIELDS},
        )

        # If existing has outcome data, remove those keys from $set
        if existing:
            for field in PRESERVE_FIELDS:
                if existing.get(field) is not None:
                    doc.pop(field, None)

        result = await db.claims.update_one(
            {"claim_id": claim.claim_id},
            {"$set": doc},
            upsert=True,
        )
        if result.upserted_id or result.modified_count:
            inserted += 1
    return inserted


async def get_claims(
    skip: int = 0,
    limit: int = 50,
    risk_level: str | None = None,
    payer_id: str | None = None,
    sort_by: str = "created_at",
    sort_order: int = -1,
) -> tuple[list[ClaimResponse], int]:
    """Get paginated claims list with optional filters. Joins prediction data."""
    db = get_db()

    # Build filter
    query: dict = {}
    if payer_id:
        query["payer_id"] = payer_id

    # If filtering by risk_level, we need to join with predictions
    pipeline = []

    if query:
        pipeline.append({"$match": query})

    # Left join predictions
    pipeline.extend([
        {
            "$lookup": {
                "from": "predictions",
                "localField": "claim_id",
                "foreignField": "claim_id",
                "as": "prediction",
            }
        },
        {"$unwind": {"path": "$prediction", "preserveNullAndEmptyArrays": True}},
    ])

    if risk_level:
        pipeline.append({"$match": {"prediction.risk_level": risk_level}})

    # Count total before pagination
    count_pipeline = pipeline + [{"$count": "total"}]
    count_result = await db.claims.aggregate(count_pipeline).to_list(1)
    total = count_result[0]["total"] if count_result else 0

    # Sort and paginate
    sort_field = sort_by
    if sort_by == "risk_score":
        sort_field = "prediction.risk_score"
    pipeline.extend([
        {"$sort": {sort_field: sort_order}},
        {"$skip": skip},
        {"$limit": limit},
    ])

    docs = await db.claims.aggregate(pipeline).to_list(limit)

    claims = []
    for doc in docs:
        pred = doc.get("prediction")
        claims.append(ClaimResponse(
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
            actual_outcome=doc.get("actual_outcome"),
            paid_amount=doc.get("paid_amount"),
            carc_codes=doc.get("carc_codes"),
        ))

    return claims, total


async def get_claim(claim_id: str) -> ClaimResponse | None:
    """Get single claim with prediction and remittance data."""
    db = get_db()

    doc = await db.claims.find_one({"claim_id": claim_id})
    if not doc:
        return None

    pred = await db.predictions.find_one({"claim_id": claim_id})
    remit = await db.remittances.find_one({"claim_id": claim_id})

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
        actual_outcome=remit.get("claim_status") if remit else doc.get("actual_outcome"),
        paid_amount=remit.get("paid_amount") if remit else doc.get("paid_amount"),
        carc_codes=remit.get("carc_codes") if remit else doc.get("carc_codes"),
    )
