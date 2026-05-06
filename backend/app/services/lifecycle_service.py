"""
Claim Lifecycle Tracking Service.

Tracks each claim's full journey: original submission → denial → fix →
resubmission → outcome.  Each claim_id gets a single `claim_lifecycle`
document with an `attempts` array so we can compute first-pass payment
rates, average attempts to payment, and resubmission success rates.
"""

from __future__ import annotations

from datetime import datetime

import structlog

from app.db.mongodb import get_db

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patient_name(claim) -> str:
    """Build 'LAST, FIRST' from a ParsedClaim (or dict)."""
    if hasattr(claim, "patient_last_name"):
        last = claim.patient_last_name or ""
        first = claim.patient_first_name or ""
    else:
        last = claim.get("patient_last_name", "")
        first = claim.get("patient_first_name", "")
    parts = [p for p in (last, first) if p]
    return ", ".join(parts) if parts else "UNKNOWN"


def _claim_field(claim, field: str, default=""):
    """Read a field from a ParsedClaim (Pydantic) or dict."""
    if hasattr(claim, field):
        return getattr(claim, field, default)
    return claim.get(field, default)


def _service_lines_summary(claim) -> list[dict]:
    """Extract compact service-line info for lifecycle storage."""
    slines = _claim_field(claim, "service_lines", [])
    result = []
    for sl in slines:
        if hasattr(sl, "model_dump"):
            result.append(sl.model_dump())
        elif isinstance(sl, dict):
            result.append(sl)
    return result


# ---------------------------------------------------------------------------
# detect_fix_applied — diff current claim vs previous attempt
# ---------------------------------------------------------------------------

def detect_fix_applied(claim, previous_attempt: dict) -> str:
    """Compare a new claim submission to the previous attempt and describe
    what changed.  Returns a human-readable string or empty string."""
    changes: list[str] = []
    prev_features = previous_attempt.get("features", {})

    # --- Modifier changes per service line ---
    prev_slines = previous_attempt.get("service_lines", [])
    curr_slines = _service_lines_summary(claim)

    prev_mods_by_cpt: dict[str, set] = {}
    for sl in prev_slines:
        cpt = sl.get("cpt_code", "")
        prev_mods_by_cpt.setdefault(cpt, set()).update(sl.get("modifiers", []))

    curr_mods_by_cpt: dict[str, set] = {}
    for sl in curr_slines:
        cpt = sl.get("cpt_code", "")
        curr_mods_by_cpt.setdefault(cpt, set()).update(sl.get("modifiers", []))

    all_cpts = set(prev_mods_by_cpt) | set(curr_mods_by_cpt)
    for cpt in sorted(all_cpts):
        prev_m = prev_mods_by_cpt.get(cpt, set())
        curr_m = curr_mods_by_cpt.get(cpt, set())
        added = curr_m - prev_m
        removed = prev_m - curr_m
        if added:
            changes.append(f"Added modifier {', '.join(sorted(added))} to {cpt}")
        if removed:
            changes.append(f"Removed modifier {', '.join(sorted(removed))} from {cpt}")

    # --- Diagnosis code changes ---
    prev_dx = set(prev_features.get("diagnosis_codes", []))
    curr_dx = set(_claim_field(claim, "diagnosis_codes", []))
    added_dx = curr_dx - prev_dx
    removed_dx = prev_dx - curr_dx
    if added_dx:
        changes.append(f"Added diagnosis {', '.join(sorted(added_dx))}")
    if removed_dx:
        changes.append(f"Removed diagnosis {', '.join(sorted(removed_dx))}")

    # --- Prior auth added ---
    prev_auth = prev_features.get("prior_auth_number", "")
    curr_auth = _claim_field(claim, "prior_auth_number", "")
    if curr_auth and not prev_auth:
        changes.append("Added prior authorization")
    elif prev_auth and not curr_auth:
        changes.append("Removed prior authorization")

    # --- Charge changes ---
    prev_charge = prev_features.get("total_charge", 0.0)
    curr_charge = float(_claim_field(claim, "total_charge", 0.0))
    if prev_charge and curr_charge and abs(curr_charge - prev_charge) > 0.01:
        changes.append(f"Changed total charge from {prev_charge} to {curr_charge}")

    return "; ".join(changes)


# ---------------------------------------------------------------------------
# create_or_update_lifecycle
# ---------------------------------------------------------------------------

async def create_or_update_lifecycle(
    claim,
    prediction,
    validation_issues: list[dict],
    fixes_recommended: list[str],
) -> None:
    """Create or append to a claim's lifecycle document.

    - First submission: creates the lifecycle doc with attempt 1.
    - Resubmission (doc exists OR frequency_code == '7'): appends a new
      attempt with fix_applied diff.
    """
    db = get_db()
    claim_id = _claim_field(claim, "claim_id")
    if not claim_id:
        return

    now = datetime.utcnow()
    freq_code = _claim_field(claim, "frequency_code", "1")
    payer_name = _claim_field(claim, "payer_name", "")
    patient_name = _patient_name(claim)

    # Feature snapshot for diffing later
    features = {
        "total_charge": float(_claim_field(claim, "total_charge", 0.0)),
        "diagnosis_codes": list(_claim_field(claim, "diagnosis_codes", [])),
        "prior_auth_number": _claim_field(claim, "prior_auth_number", ""),
        "place_of_service": _claim_field(claim, "place_of_service", ""),
        "frequency_code": freq_code,
    }

    service_lines = _service_lines_summary(claim)

    # Prediction info
    risk_score = 0.0
    risk_level = "LOW"
    model_version = "unknown"
    if prediction:
        if hasattr(prediction, "risk_score"):
            risk_score = prediction.risk_score
            risk_level = prediction.risk_level or "LOW"
            model_version = getattr(prediction, "model_version", "unknown") or "unknown"
        elif isinstance(prediction, dict):
            risk_score = prediction.get("risk_score", 0.0)
            risk_level = prediction.get("risk_level", "LOW")
            model_version = prediction.get("model_version", "unknown")

    billed_amount = float(_claim_field(claim, "total_charge", 0.0))

    # Check if lifecycle already exists
    existing = await db.claim_lifecycle.find_one({"claim_id": claim_id})

    if existing:
        # --- Resubmission: append new attempt ---
        prev_attempt = existing["attempts"][-1] if existing.get("attempts") else {}
        fix_applied = detect_fix_applied(claim, prev_attempt) if prev_attempt else ""
        attempt_number = existing.get("total_attempts", len(existing.get("attempts", []))) + 1

        new_attempt = {
            "attempt_number": attempt_number,
            "submitted_at": now,
            "frequency_code": freq_code,
            "fix_applied": fix_applied,
            "features": features,
            "service_lines": service_lines,
            "prediction_risk_score": risk_score,
            "prediction_risk_level": risk_level,
            "validation_issues": validation_issues,
            "fixes_recommended": fixes_recommended,
            "status": "PENDING",
            "denial_codes": [],
            "paid_amount": 0.0,
            "billed_amount": billed_amount,
            "remittance_date": None,
            "model_version": model_version,
        }

        await db.claim_lifecycle.update_one(
            {"claim_id": claim_id},
            {
                "$push": {"attempts": new_attempt},
                "$set": {
                    "total_attempts": attempt_number,
                    "current_status": "PENDING",
                    "updated_at": now,
                    "payer_name": payer_name,
                    "patient_name": patient_name,
                },
            },
        )
        logger.info(
            "Lifecycle updated — resubmission",
            claim_id=claim_id,
            attempt=attempt_number,
            fix_applied=fix_applied or "(none detected)",
        )
    else:
        # --- First submission ---
        first_attempt = {
            "attempt_number": 1,
            "submitted_at": now,
            "frequency_code": freq_code,
            "features": features,
            "service_lines": service_lines,
            "prediction_risk_score": risk_score,
            "prediction_risk_level": risk_level,
            "validation_issues": validation_issues,
            "fixes_recommended": fixes_recommended,
            "status": "PENDING",
            "denial_codes": [],
            "paid_amount": 0.0,
            "billed_amount": billed_amount,
            "remittance_date": None,
            "model_version": model_version,
        }

        lifecycle_doc = {
            "claim_id": claim_id,
            "payer_name": payer_name,
            "patient_name": patient_name,
            "current_status": "PENDING",
            "total_attempts": 1,
            "created_at": now,
            "updated_at": now,
            "attempts": [first_attempt],
        }

        await db.claim_lifecycle.insert_one(lifecycle_doc)
        logger.info("Lifecycle created", claim_id=claim_id)


# ---------------------------------------------------------------------------
# update_lifecycle_outcome — called when 835 arrives
# ---------------------------------------------------------------------------

async def update_lifecycle_outcome(claim_id: str, remit, claim_doc: dict | None = None) -> dict | None:
    """Update the latest PENDING attempt with remittance outcome.

    If no lifecycle exists yet (e.g. claim submitted before lifecycle
    feature was added, or submitted outside this system), creates one
    from the matched claim doc so the denial/payment is still tracked.

    Returns the updated lifecycle doc (or None if not found).
    """
    db = get_db()
    lifecycle = await db.claim_lifecycle.find_one({"claim_id": claim_id})

    if not lifecycle:
        # Backfill: create lifecycle from the claim doc + remittance
        if not claim_doc:
            claim_doc = await db.claims.find_one({"claim_id": claim_id})
        if not claim_doc:
            return None

        now = datetime.utcnow()
        payer_name = claim_doc.get("payer_name", "")
        last = claim_doc.get("patient_last_name", "")
        first = claim_doc.get("patient_first_name", "")
        patient_name = ", ".join(p for p in (last, first) if p) or "UNKNOWN"

        # Determine status from remit
        raw_status = remit.claim_status if hasattr(remit, "claim_status") else remit.get("claim_status", "")
        if raw_status == "paid":
            status = "PAID"
        elif raw_status == "denied":
            status = "DENIED"
        else:
            status = "PARTIAL"

        paid_amount = (remit.paid_amount if hasattr(remit, "paid_amount") else remit.get("paid_amount", 0.0)) or 0.0
        billed_amount = (remit.billed_amount if hasattr(remit, "billed_amount") else remit.get("billed_amount", 0.0)) or 0.0
        denial_codes = list((remit.carc_codes if hasattr(remit, "carc_codes") else remit.get("carc_codes", [])) or [])

        # Reconstruct features from claim doc
        slines = claim_doc.get("service_lines", [])
        features = {
            "total_charge": float(claim_doc.get("total_charge", 0.0)),
            "diagnosis_codes": list(claim_doc.get("diagnosis_codes", [])),
            "prior_auth_number": claim_doc.get("prior_auth_number", ""),
            "place_of_service": claim_doc.get("place_of_service", ""),
            "frequency_code": claim_doc.get("frequency_code", "1"),
        }

        # Pull prediction if available
        prediction = await db.predictions.find_one({"claim_id": claim_id})
        risk_score = prediction.get("risk_score", 0.0) if prediction else 0.0
        risk_level = prediction.get("risk_level", "LOW") if prediction else "LOW"
        model_version = prediction.get("model_version", "unknown") if prediction else "unknown"

        backfilled_attempt = {
            "attempt_number": 1,
            "submitted_at": claim_doc.get("created_at", now),
            "frequency_code": claim_doc.get("frequency_code", "1"),
            "features": features,
            "service_lines": slines,
            "prediction_risk_score": risk_score,
            "prediction_risk_level": risk_level,
            "validation_issues": claim_doc.get("validation_issues", []),
            "fixes_recommended": [],
            "status": status,
            "denial_codes": denial_codes,
            "paid_amount": paid_amount,
            "billed_amount": billed_amount or float(claim_doc.get("total_charge", 0.0)),
            "remittance_date": now,
            "model_version": model_version,
        }

        lifecycle_doc = {
            "claim_id": claim_id,
            "payer_name": payer_name,
            "patient_name": patient_name,
            "current_status": status,
            "total_attempts": 1,
            "created_at": claim_doc.get("created_at", now),
            "updated_at": now,
            "attempts": [backfilled_attempt],
        }

        await db.claim_lifecycle.insert_one(lifecycle_doc)
        logger.info(
            "Lifecycle backfilled from 835",
            claim_id=claim_id,
            status=status,
        )
        return await db.claim_lifecycle.find_one({"claim_id": claim_id})

    now = datetime.utcnow()
    attempts = lifecycle.get("attempts", [])

    # Find the latest PENDING attempt
    target_idx = None
    for i in reversed(range(len(attempts))):
        if attempts[i].get("status") == "PENDING":
            target_idx = i
            break

    if target_idx is None:
        logger.debug("No PENDING attempt to update", claim_id=claim_id)
        return lifecycle

    # Determine status from remittance
    claim_status = ""
    paid_amount = 0.0
    billed_amount = 0.0
    denial_codes: list[str] = []

    if hasattr(remit, "claim_status"):
        claim_status = remit.claim_status
        paid_amount = remit.paid_amount or 0.0
        billed_amount = remit.billed_amount or 0.0
        denial_codes = list(remit.carc_codes or [])
    elif isinstance(remit, dict):
        claim_status = remit.get("claim_status", "")
        paid_amount = remit.get("paid_amount", 0.0)
        billed_amount = remit.get("billed_amount", 0.0)
        denial_codes = list(remit.get("carc_codes", []))

    if claim_status == "paid":
        status = "PAID"
    elif claim_status == "denied":
        status = "DENIED"
    else:
        status = "PARTIAL"

    # Handle void (frequency_code "8")
    freq = attempts[target_idx].get("frequency_code", "")
    if freq == "8":
        status = "VOID"

    # Build $set for the specific array element
    prefix = f"attempts.{target_idx}"
    update_fields = {
        f"{prefix}.status": status,
        f"{prefix}.denial_codes": denial_codes,
        f"{prefix}.paid_amount": paid_amount,
        f"{prefix}.billed_amount": billed_amount or attempts[target_idx].get("billed_amount", 0.0),
        f"{prefix}.remittance_date": now,
        "current_status": status,
        "updated_at": now,
    }

    await db.claim_lifecycle.update_one(
        {"claim_id": claim_id},
        {"$set": update_fields},
    )

    logger.info(
        "Lifecycle outcome updated",
        claim_id=claim_id,
        attempt=target_idx + 1,
        status=status,
    )

    # Return refreshed doc
    return await db.claim_lifecycle.find_one({"claim_id": claim_id})


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

async def get_lifecycle(claim_id: str) -> dict | None:
    """Get full lifecycle for a single claim."""
    db = get_db()
    doc = await db.claim_lifecycle.find_one({"claim_id": claim_id})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def get_lifecycles(
    skip: int = 0,
    limit: int = 50,
    status: str | None = None,
    min_attempts: int | None = None,
    payer_name: str | None = None,
) -> tuple[list[dict], int]:
    """Paginated lifecycle list with optional filters."""
    db = get_db()
    query: dict = {}
    if status:
        query["current_status"] = status.upper()
    if min_attempts is not None and min_attempts > 0:
        query["total_attempts"] = {"$gte": min_attempts}
    if payer_name:
        query["payer_name"] = {"$regex": payer_name, "$options": "i"}

    total = await db.claim_lifecycle.count_documents(query)
    docs = (
        await db.claim_lifecycle.find(query)
        .sort("updated_at", -1)
        .skip(skip)
        .limit(limit)
        .to_list(limit)
    )

    for doc in docs:
        doc["_id"] = str(doc["_id"])

    return docs, total


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

async def get_lifecycle_stats() -> dict:
    """Compute aggregate lifecycle statistics."""
    db = get_db()

    total = await db.claim_lifecycle.count_documents({})
    if total == 0:
        return {
            "total_claims_tracked": 0,
            "first_pass_payment_rate": 0.0,
            "avg_attempts_to_payment": 0.0,
            "resubmission_success_rate": 0.0,
            "status_breakdown": {},
            "payer_breakdown": [],
        }

    # --- First-pass payment rate ---
    # Claims where attempt 1 resulted in PAID
    first_pass_paid_pipeline = [
        {"$match": {"attempts.0.status": {"$in": ["PAID", "PARTIAL"]}}},
        {"$project": {"first_paid": {"$cond": [
            {"$eq": [{"$arrayElemAt": ["$attempts.status", 0]}, "PAID"]},
            1, 0,
        ]}}},
        {"$group": {"_id": None, "count": {"$sum": "$first_paid"}}},
    ]
    first_pass_result = await db.claim_lifecycle.aggregate(first_pass_paid_pipeline).to_list(1)
    first_pass_paid = first_pass_result[0]["count"] if first_pass_result else 0

    # Total claims that have a resolved first attempt
    resolved_first_pipeline = [
        {"$match": {"attempts.0.status": {"$in": ["PAID", "DENIED", "PARTIAL"]}}},
        {"$count": "total"},
    ]
    resolved_first_result = await db.claim_lifecycle.aggregate(resolved_first_pipeline).to_list(1)
    resolved_first = resolved_first_result[0]["total"] if resolved_first_result else 0

    first_pass_rate = round(first_pass_paid / resolved_first, 4) if resolved_first else 0.0

    # --- Avg attempts to payment ---
    avg_attempts_pipeline = [
        {"$match": {"current_status": "PAID"}},
        {"$group": {"_id": None, "avg": {"$avg": "$total_attempts"}}},
    ]
    avg_result = await db.claim_lifecycle.aggregate(avg_attempts_pipeline).to_list(1)
    avg_attempts = round(avg_result[0]["avg"], 2) if avg_result else 0.0

    # --- Resubmission success rate ---
    # Claims with >1 attempt that eventually got PAID
    resub_total_pipeline = [
        {"$match": {"total_attempts": {"$gt": 1}}},
        {"$count": "total"},
    ]
    resub_paid_pipeline = [
        {"$match": {"total_attempts": {"$gt": 1}, "current_status": "PAID"}},
        {"$count": "total"},
    ]
    resub_total_result = await db.claim_lifecycle.aggregate(resub_total_pipeline).to_list(1)
    resub_paid_result = await db.claim_lifecycle.aggregate(resub_paid_pipeline).to_list(1)
    resub_total = resub_total_result[0]["total"] if resub_total_result else 0
    resub_paid = resub_paid_result[0]["total"] if resub_paid_result else 0
    resub_rate = round(resub_paid / resub_total, 4) if resub_total else 0.0

    # --- Status breakdown ---
    status_pipeline = [
        {"$group": {"_id": "$current_status", "count": {"$sum": 1}}},
    ]
    status_result = await db.claim_lifecycle.aggregate(status_pipeline).to_list(20)
    status_breakdown = {r["_id"]: r["count"] for r in status_result}

    # --- Payer breakdown ---
    payer_pipeline = [
        {"$group": {
            "_id": "$payer_name",
            "total": {"$sum": 1},
            "paid": {"$sum": {"$cond": [{"$eq": ["$current_status", "PAID"]}, 1, 0]}},
            "denied": {"$sum": {"$cond": [{"$eq": ["$current_status", "DENIED"]}, 1, 0]}},
            "pending": {"$sum": {"$cond": [{"$eq": ["$current_status", "PENDING"]}, 1, 0]}},
            "avg_attempts": {"$avg": "$total_attempts"},
        }},
        {"$sort": {"total": -1}},
        {"$limit": 20},
    ]
    payer_result = await db.claim_lifecycle.aggregate(payer_pipeline).to_list(20)
    payer_breakdown = [
        {
            "payer_name": r["_id"],
            "total": r["total"],
            "paid": r["paid"],
            "denied": r["denied"],
            "pending": r["pending"],
            "avg_attempts": round(r["avg_attempts"], 2),
            "payment_rate": round(r["paid"] / r["total"], 4) if r["total"] else 0.0,
        }
        for r in payer_result
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
