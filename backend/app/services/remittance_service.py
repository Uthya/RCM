"""CRUD operations for parsed 835 remittance data."""

from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import (
    remittance_repo, claim_repo, outcome_repo, prediction_repo, training_repo,
    lifecycle_repo,
)
from app.schemas.remittance import ParsedRemittance, RemittanceResponse
from app.services.claim_rules import classify_issue_type

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
    return [
        {"code": code, "description": CARC_DESCRIPTIONS.get(code, "Unknown adjustment reason")}
        for code in carc_codes
    ]


async def _create_training_record(
    session: AsyncSession,
    claim_id: str,
    remit,
    claim: dict,
    attempt_number: int = 1,
) -> bool:
    """Join prediction features with 835 outcome into ml_training_data.

    Creates one record per (claim_id, attempt_number) pair.
    Returns True if a record was created, False otherwise.
    """
    existing = await training_repo.find_existing(session, claim_id, attempt_number=attempt_number)
    if existing:
        logger.debug("Training record already exists for attempt", claim_id=claim_id, attempt_number=attempt_number)
        return False

    prediction = await prediction_repo.find_prediction(session, claim_id)
    if not prediction or not prediction.get("features"):
        logger.debug("No prediction features for training join", claim_id=claim_id)
        return False

    label = 1 if remit.claim_status == "denied" else 0
    first_carc = remit.carc_codes[0] if remit.carc_codes else None

    from app.core.feature_engineer import FEATURE_VERSION, FEATURE_COUNT, FEATURE_HASH

    training_doc = {
        "claim_id": claim_id,
        "attempt_number": attempt_number,
        "is_first_attempt": (attempt_number == 1),
        "features": prediction["features"],
        "label": label,
        "actual_outcome": remit.claim_status,
        "denial_code": first_carc,
        "denial_code_description": CARC_DESCRIPTIONS.get(first_carc, "Unknown") if first_carc else None,
        "all_carc_codes": remit.carc_codes,
        "paid_amount": remit.paid_amount,
        "billed_amount": remit.billed_amount,
        "model_version_at_prediction": prediction.get("model_version", "unknown"),
        "prediction_risk_score": prediction.get("risk_score"),
        "feature_version": prediction.get("feature_version", FEATURE_VERSION),
        "feature_count": FEATURE_COUNT,
        "feature_hash": FEATURE_HASH,
        "created_at": datetime.utcnow(),
    }

    await training_repo.insert_record(session, training_doc)
    logger.info("Training record created", claim_id=claim_id, attempt_number=attempt_number, label=label)
    return True


async def store_remittances(session: AsyncSession, remittances: list[ParsedRemittance]) -> dict:
    """Insert parsed 835 remittance records. Returns summary."""
    inserted = 0
    matched = 0
    denied = 0
    new_outcomes = 0
    training_records_created = 0

    for remit in remittances:
        doc = remit.model_dump()
        doc["created_at"] = datetime.utcnow()

        await remittance_repo.insert_remittance(session, doc)
        inserted += 1

        if remit.claim_status in ("denied",):
            denied += 1

        claim = await claim_repo.find_claim(session, remit.claim_id)
        if claim:
            # Resolve which attempt this 835 belongs to
            attempt_number, attempt_type = await lifecycle_repo.get_pending_attempt_info(
                session, remit.claim_id,
            )

            existing_outcome = await outcome_repo.find_outcome(
                session, remit.claim_id, attempt_number=attempt_number,
            )
            if not existing_outcome:
                new_outcomes += 1

            carc_descs = _enrich_carc_descriptions(remit.carc_codes)
            is_new = await outcome_repo.upsert_outcome(session, {
                "claim_id": remit.claim_id,
                "attempt_number": attempt_number,
                "attempt_type": attempt_type,
                "outcome_status": remit.claim_status,
                "paid_amount": remit.paid_amount,
                "carc_codes": remit.carc_codes,
                "carc_descriptions": carc_descs,
                "created_at": datetime.utcnow(),
            })
            matched += 1
            logger.info("Matched 835 to claim", claim_id=remit.claim_id, status=remit.claim_status,
                        attempt_number=attempt_number)

            # Training data join
            try:
                created = await _create_training_record(
                    session, remit.claim_id, remit, claim,
                    attempt_number=attempt_number,
                )
                if created:
                    training_records_created += 1
            except Exception as e:
                logger.warning("Failed to create training record", claim_id=remit.claim_id, error=str(e))

            # Lifecycle outcome update
            lifecycle_doc = None
            try:
                from app.services.lifecycle_service import update_lifecycle_outcome
                lifecycle_doc = await update_lifecycle_outcome(session, remit.claim_id, remit, claim_doc=claim)
            except Exception as e:
                logger.warning("Failed to update lifecycle outcome", claim_id=remit.claim_id, error=str(e))

            # Knowledge layer: record fix outcomes with relevance filtering
            try:
                from app.services.knowledge_store import (
                    record_fix, CARC_FIX_MAP, filter_changes_for_issue,
                )
                payer = claim.get("payer_name", "")
                slines = claim.get("service_lines", [])
                cpt = slines[0].get("cpt_code", "") if slines else ""

                is_resubmission = (
                    lifecycle_doc
                    and lifecycle_doc.get("total_attempts", 1) > 1
                )

                if is_resubmission:
                    attempts = lifecycle_doc.get("attempts", [])
                    latest = attempts[-1] if attempts else {}
                    first = attempts[0] if attempts else {}
                    fix_applied_text = latest.get("fix_applied", "")
                    fix_changes = latest.get("features", {}).get("_fix_changes", {})

                    if fix_applied_text:
                        first_issues = first.get("validation_issues", [])
                        for iss in first_issues:
                            issue_type = classify_issue_type(iss.get("reason", ""))
                            if not issue_type:
                                continue
                            # Filter to only relevant changes for this issue type
                            relevant_text = (
                                filter_changes_for_issue(fix_changes, issue_type)
                                if fix_changes else fix_applied_text
                            )
                            if not relevant_text:
                                continue
                            await record_fix(
                                session,
                                claim_id=remit.claim_id,
                                issue_type=issue_type,
                                fix_applied=relevant_text,
                                payer_name=payer,
                                cpt_code=cpt,
                                outcome=remit.claim_status,
                                attempt_number=attempt_number,
                            )

                    # CARC correlation: map previous denial codes to filtered fix outcomes
                    prev_denied = [
                        a for a in attempts[:-1]
                        if a.get("status") == "DENIED" and a.get("denial_codes")
                    ]
                    if prev_denied:
                        prev_denial_codes = prev_denied[-1].get("denial_codes", [])
                        for code in prev_denial_codes:
                            carc_issue_type = CARC_FIX_MAP.get(code)
                            if not carc_issue_type:
                                continue
                            relevant_text = (
                                filter_changes_for_issue(fix_changes, carc_issue_type)
                                if fix_changes else (fix_applied_text or "Resubmission (no field changes detected)")
                            )
                            if not relevant_text:
                                continue
                            await record_fix(
                                session,
                                claim_id=remit.claim_id,
                                issue_type=carc_issue_type,
                                fix_applied=relevant_text,
                                payer_name=payer,
                                cpt_code=cpt,
                                outcome=remit.claim_status,
                                attempt_number=attempt_number,
                            )
                # Paid first-attempts are NOT recorded — no fix was actually applied
            except Exception as e:
                logger.warning("Failed to record fix", error=str(e))

    total_matched = await outcome_repo.count_outcomes(session)
    total_training = await training_repo.count_records(session)

    return {
        "inserted": inserted,
        "matched": matched,
        "denied": denied,
        "new_outcomes": new_outcomes,
        "total_paid": sum(r.paid_amount for r in remittances),
        "total_matched_claims": total_matched,
        "training_records_created": training_records_created,
        "total_training_records": total_training,
    }


async def get_remittances(
    session: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[RemittanceResponse], int]:
    rows, total = await remittance_repo.get_remittances_paginated(session, skip=skip, limit=limit)

    remittances = []
    for doc in rows:
        remittances.append(RemittanceResponse(
            id=str(doc["id"]),
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


async def get_remittance(session: AsyncSession, remittance_id: str) -> RemittanceResponse | None:
    """Get single remittance by integer PK."""
    try:
        rid = int(remittance_id)
    except (ValueError, TypeError):
        return None

    doc = await remittance_repo.get_remittance_by_id(session, rid)
    if not doc:
        return None

    return RemittanceResponse(
        id=str(doc["id"]),
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
