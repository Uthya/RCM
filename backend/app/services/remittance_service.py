"""CRUD operations for parsed 835 remittance data in MongoDB."""

from datetime import datetime

import structlog

from app.db.mongodb import get_db
from app.schemas.remittance import ParsedRemittance, RemittanceResponse

logger = structlog.get_logger()

# ── Top 25 CARC code descriptions ──
CARC_DESCRIPTIONS: dict[str, str] = {
    "1": "Deductible amount",
    "2": "Coinsurance amount",
    "3": "Co-payment amount",
    "4": "The procedure code is inconsistent with the modifier used",
    "5": "The procedure code/bill type is inconsistent with the place of service",
    "6": "The procedure/revenue code is inconsistent with the patient's age",
    "9": "The diagnosis is inconsistent with the patient's age",
    "11": "The diagnosis is inconsistent with the procedure",
    "16": "Claim/service lacks information or has submission/billing error(s)",
    "18": "Exact duplicate claim/service",
    "22": "This care may be covered by another payer per coordination of benefits",
    "23": "The impact of prior payer(s) adjudication including payments and/or adjustments",
    "27": "Expenses incurred after coverage terminated",
    "29": "The time limit for filing has expired",
    "31": "Patient cannot be identified as our insured",
    "35": "Lifetime benefit maximum has been reached",
    "45": "Charge exceeds fee schedule/maximum allowable or contracted/legislated fee arrangement",
    "50": "These are non-covered services because this is not deemed a medical necessity",
    "96": "Non-covered charge(s). At least one Remark Code must be provided",
    "97": "The benefit for this service is included in the payment/allowance for another service",
    "109": "Claim/service not covered by this payer/contractor",
    "119": "Benefit maximum for this time period or occurrence has been reached",
    "167": "This (these) diagnosis(es) is (are) not covered",
    "197": "Precertification/authorization/notification/pre-treatment absent",
    "204": "This service/equipment/drug is not covered under the patient's current benefit plan",
}


def _enrich_carc_descriptions(carc_codes: list[str]) -> list[dict]:
    """Return list of {code, description} for known CARC codes."""
    return [
        {"code": code, "description": CARC_DESCRIPTIONS.get(code, "Unknown adjustment reason")}
        for code in carc_codes
    ]


async def store_remittances(remittances: list[ParsedRemittance]) -> dict:
    """Insert parsed 835 remittance records. Returns summary."""
    db = get_db()
    inserted = 0
    matched = 0
    denied = 0
    new_outcomes = 0

    for remit in remittances:
        doc = remit.model_dump()
        doc["created_at"] = datetime.utcnow()

        await db.remittances.insert_one(doc)
        inserted += 1

        if remit.claim_status in ("denied",):
            denied += 1

        # Try to match to existing claim and update outcome
        claim = await db.claims.find_one({"claim_id": remit.claim_id})
        if claim:
            # Check if this is a new outcome (wasn't matched before)
            if not claim.get("actual_outcome"):
                new_outcomes += 1

            carc_descs = _enrich_carc_descriptions(remit.carc_codes)
            await db.claims.update_one(
                {"claim_id": remit.claim_id},
                {"$set": {
                    "actual_outcome": remit.claim_status,
                    "paid_amount": remit.paid_amount,
                    "carc_codes": remit.carc_codes,
                    "carc_descriptions": carc_descs,
                }},
            )
            matched += 1
            logger.info("Matched 835 to claim", claim_id=remit.claim_id, status=remit.claim_status)

            # ── Knowledge layer: record fix outcomes ──
            validation_issues = claim.get("validation_issues", [])
            if validation_issues:
                try:
                    from app.services.knowledge_store import record_fix
                    payer = claim.get("payer_name", "")
                    slines = claim.get("service_lines", [])
                    cpt = slines[0].get("cpt_code", "") if slines else ""
                    for iss in validation_issues:
                        reason_first = iss.get("reason", "").split("\n")[0].lower()
                        if "missing modifier" in reason_first:
                            issue_type = "missing_modifier"
                        elif "invalid cpt" in reason_first:
                            issue_type = "invalid_cpt"
                        elif "prior authorization" in reason_first:
                            issue_type = "missing_prior_auth"
                        else:
                            continue
                        fixes = iss.get("fixes", [])
                        fix_text = fixes[0] if fixes else "unknown"
                        await record_fix(
                            claim_id=remit.claim_id,
                            issue_type=issue_type,
                            fix_applied=fix_text,
                            payer_name=payer,
                            cpt_code=cpt,
                            outcome=remit.claim_status,
                        )
                except Exception as e:
                    logger.warning("Failed to record fix", error=str(e))

    total_matched = await db.claims.count_documents({"actual_outcome": {"$exists": True}})

    return {
        "inserted": inserted,
        "matched": matched,
        "denied": denied,
        "new_outcomes": new_outcomes,
        "total_paid": sum(r.paid_amount for r in remittances),
        "total_matched_claims": total_matched,
    }


async def get_remittances(
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[RemittanceResponse], int]:
    """Get paginated remittance list."""
    db = get_db()

    total = await db.remittances.count_documents({})
    docs = await db.remittances.find().sort("created_at", -1).skip(skip).limit(limit).to_list(limit)

    remittances = []
    for doc in docs:
        remittances.append(RemittanceResponse(
            id=str(doc["_id"]),
            claim_id=doc.get("claim_id", ""),
            payer_control_number=doc.get("payer_control_number", ""),
            claim_status=doc.get("claim_status", ""),
            billed_amount=doc.get("billed_amount", 0.0),
            paid_amount=doc.get("paid_amount", 0.0),
            patient_responsibility=doc.get("patient_responsibility", 0.0),
            payer_name=doc.get("payer_name", ""),
            payee_name=doc.get("payee_name", ""),
            trace_number=doc.get("trace_number", ""),
            payment_date=doc.get("payment_date", ""),
            carc_codes=doc.get("carc_codes", []),
            rarc_codes=doc.get("rarc_codes", []),
            adjustments=doc.get("adjustments", []),
            service_lines=doc.get("service_lines", []),
            created_at=doc.get("created_at"),
        ))

    return remittances, total


async def get_remittance(remittance_id: str) -> RemittanceResponse | None:
    """Get single remittance by MongoDB _id."""
    from bson import ObjectId
    db = get_db()

    doc = await db.remittances.find_one({"_id": ObjectId(remittance_id)})
    if not doc:
        return None

    return RemittanceResponse(
        id=str(doc["_id"]),
        claim_id=doc.get("claim_id", ""),
        payer_control_number=doc.get("payer_control_number", ""),
        claim_status=doc.get("claim_status", ""),
        billed_amount=doc.get("billed_amount", 0.0),
        paid_amount=doc.get("paid_amount", 0.0),
        patient_responsibility=doc.get("patient_responsibility", 0.0),
        payer_name=doc.get("payer_name", ""),
        payee_name=doc.get("payee_name", ""),
        trace_number=doc.get("trace_number", ""),
        payment_date=doc.get("payment_date", ""),
        carc_codes=doc.get("carc_codes", []),
        rarc_codes=doc.get("rarc_codes", []),
        adjustments=doc.get("adjustments", []),
        service_lines=doc.get("service_lines", []),
        created_at=doc.get("created_at"),
    )
