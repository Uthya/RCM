"""
Claim Lifecycle Tracking Service.

Tracks each claim's full journey: original submission -> denial -> fix ->
resubmission -> outcome. Uses normalized lifecycle_attempts table.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import lifecycle_repo, claim_repo, prediction_repo

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patient_name(claim) -> str:
    if hasattr(claim, "patient_last_name"):
        last = claim.patient_last_name or ""
        first = claim.patient_first_name or ""
    else:
        last = claim.get("patient_last_name", "")
        first = claim.get("patient_first_name", "")
    parts = [p for p in (last, first) if p]
    return ", ".join(parts) if parts else "UNKNOWN"


def _claim_field(claim, field: str, default=""):
    if hasattr(claim, field):
        return getattr(claim, field, default)
    return claim.get(field, default)


def _service_lines_summary(claim) -> list[dict]:
    slines = _claim_field(claim, "service_lines", [])
    result = []
    for sl in slines:
        if hasattr(sl, "model_dump"):
            result.append(sl.model_dump())
        elif isinstance(sl, dict):
            result.append(sl)
    return result


def _serialize_claim_snapshot(claim) -> dict:
    """Create an immutable, normalized snapshot of claim data for comparison.

    Both sides of detect_fix_applied() must go through this function
    to eliminate asymmetric type coercion.
    """
    slines = _service_lines_summary(claim)
    return {
        # Claim-level fields (normalized to str)
        "claim_id": str(_claim_field(claim, "claim_id", "") or ""),
        "total_charge": float(_claim_field(claim, "total_charge", 0.0) or 0.0),
        "diagnosis_codes": sorted(str(d) for d in (_claim_field(claim, "diagnosis_codes", []) or [])),
        "prior_auth_number": str(_claim_field(claim, "prior_auth_number", "") or ""),
        "place_of_service": str(_claim_field(claim, "place_of_service", "") or ""),
        "frequency_code": str(_claim_field(claim, "frequency_code", "") or ""),
        "subscriber_id": str(_claim_field(claim, "subscriber_id", "") or ""),
        "billing_provider_npi": str(_claim_field(claim, "billing_provider_npi", "") or ""),
        "rendering_provider_npi": str(_claim_field(claim, "rendering_provider_npi", "") or ""),
        "provider_taxonomy": str(_claim_field(claim, "provider_taxonomy", "") or ""),
        "payer_id": str(_claim_field(claim, "payer_id", "") or ""),
        "payer_name": str(_claim_field(claim, "payer_name", "") or ""),
        "group_number": str(_claim_field(claim, "group_number", "") or ""),
        "patient_dob": str(_claim_field(claim, "patient_dob", "") or ""),
        "patient_gender": str(_claim_field(claim, "patient_gender", "") or ""),
        # EDI envelope fields (previously lost from features dict)
        "transaction_reference": str(_claim_field(claim, "transaction_reference", "") or ""),
        "sender_id": str(_claim_field(claim, "sender_id", "") or ""),
        "receiver_id": str(_claim_field(claim, "receiver_id", "") or ""),
        "interchange_control_number": str(_claim_field(claim, "interchange_control_number", "") or ""),
        # Full service lines (normalized)
        "service_lines": slines,
    }


def _classify_attempt_type(frequency_code: str | None, attempt_number: int) -> str:
    """Classify attempt type from EDI frequency code (CLM05-3).

    Frequency code takes priority: a first transmission can still be
    a void (8) or replacement (7) in payer-initiated workflows.
    """
    if frequency_code == "8":
        return "VOID"
    if frequency_code == "7":
        return "REPLACEMENT"
    if frequency_code == "6":
        return "CORRECTED"
    if attempt_number == 1:
        return "ORIGINAL"
    return "RESUBMISSION"


# ---------------------------------------------------------------------------
# detect_fix_applied
# ---------------------------------------------------------------------------

_FIELD_LABELS = {
    "place_of_service": "place of service",
    "frequency_code": "frequency code",
    "subscriber_id": "subscriber/member ID",
    "billing_provider_npi": "billing provider NPI",
    "rendering_provider_npi": "rendering provider NPI",
    "provider_taxonomy": "provider taxonomy",
    "payer_id": "payer ID",
    "payer_name": "payer name",
    "group_number": "group number",
    "patient_dob": "patient DOB",
    "patient_gender": "patient gender",
}

_FIELD_CATEGORY = {
    "place_of_service": "place_of_service",
    "frequency_code": "frequency_code",
    "subscriber_id": "subscriber_id",
    "billing_provider_npi": "npi",
    "rendering_provider_npi": "npi",
    "provider_taxonomy": "other",
    "payer_id": "payer_id",
    "payer_name": "other",
    "group_number": "other",
    "patient_dob": "patient_dob",
    "patient_gender": "other",
}


def detect_fix_applied(claim, previous_attempt: dict) -> dict:
    """Compare current claim against previous attempt and return categorized changes.

    Uses symmetric snapshot comparison when available (_raw_snapshot),
    falling back to old asymmetric features dict for backward compatibility.

    Returns dict with category keys mapping to lists of change descriptions,
    plus a '_flat' key with all changes joined as a semicolon-separated string.
    """
    categorized: dict[str, list[str]] = {
        "modifier": [], "diagnosis": [], "prior_auth": [], "charge": [],
        "place_of_service": [], "frequency_code": [], "subscriber_id": [],
        "npi": [], "patient_dob": [], "payer_id": [], "cpt": [],
        "service_line_count": [], "units": [], "service_date": [], "other": [],
    }

    curr = _serialize_claim_snapshot(claim)

    prev_features = previous_attempt.get("features", {})
    prev_snapshot = prev_features.get("_raw_snapshot")

    if prev_snapshot:
        # Symmetric comparison: both sides from _serialize_claim_snapshot
        prev = prev_snapshot
        prev_slines = prev.get("service_lines", [])
    else:
        # Backward compat: old data without snapshot
        prev = prev_features
        prev_slines = previous_attempt.get("service_lines", [])

    curr_slines = curr.get("service_lines", [])

    # ── Modifier changes per CPT ──
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
            categorized["modifier"].append(f"Added modifier {', '.join(sorted(added))} to {cpt}")
        if removed:
            categorized["modifier"].append(f"Removed modifier {', '.join(sorted(removed))} from {cpt}")

    # ── Diagnosis code changes ──
    prev_dx = set(prev.get("diagnosis_codes", []))
    curr_dx = set(curr.get("diagnosis_codes", []))
    added_dx = curr_dx - prev_dx
    removed_dx = prev_dx - curr_dx
    if added_dx:
        categorized["diagnosis"].append(f"Added diagnosis {', '.join(sorted(added_dx))}")
    if removed_dx:
        categorized["diagnosis"].append(f"Removed diagnosis {', '.join(sorted(removed_dx))}")

    # ── Prior authorization ──
    prev_auth = str(prev.get("prior_auth_number", "") or "")
    curr_auth = str(curr.get("prior_auth_number", "") or "")
    if curr_auth and not prev_auth:
        categorized["prior_auth"].append("Added prior authorization")
    elif prev_auth and not curr_auth:
        categorized["prior_auth"].append("Removed prior authorization")
    elif prev_auth and curr_auth and prev_auth != curr_auth:
        categorized["prior_auth"].append(f"Changed prior authorization from {prev_auth} to {curr_auth}")

    # ── Total charge ──
    prev_charge = float(prev.get("total_charge", 0.0) or 0.0)
    curr_charge = float(curr.get("total_charge", 0.0) or 0.0)
    if prev_charge and curr_charge and abs(curr_charge - prev_charge) > 0.01:
        categorized["charge"].append(f"Changed total charge from {prev_charge} to {curr_charge}")

    # ── Claim-level field comparisons ──
    for field_key, label in _FIELD_LABELS.items():
        prev_val = str(prev.get(field_key, "") or "")
        curr_val = str(curr.get(field_key, "") or "")
        if prev_val != curr_val:
            cat = _FIELD_CATEGORY[field_key]
            if curr_val and not prev_val:
                categorized[cat].append(f"Added {label}: {curr_val}")
            elif prev_val and not curr_val:
                categorized[cat].append(f"Removed {label}")
            else:
                categorized[cat].append(f"Changed {label} from {prev_val} to {curr_val}")

    # ── Service-line level comparisons ──
    prev_cpts = set(sl.get("cpt_code", "") for sl in prev_slines)
    curr_cpts = set(sl.get("cpt_code", "") for sl in curr_slines)
    added_cpts = curr_cpts - prev_cpts
    removed_cpts = prev_cpts - curr_cpts
    if added_cpts:
        categorized["cpt"].append(f"Added CPT codes: {', '.join(sorted(added_cpts))}")
    if removed_cpts:
        categorized["cpt"].append(f"Removed CPT codes: {', '.join(sorted(removed_cpts))}")

    if len(prev_slines) != len(curr_slines):
        categorized["service_line_count"].append(
            f"Service line count changed from {len(prev_slines)} to {len(curr_slines)}"
        )

    # Per-line unit, charge, and date changes (match by CPT)
    prev_by_cpt: dict[str, list[dict]] = {}
    for sl in prev_slines:
        prev_by_cpt.setdefault(sl.get("cpt_code", ""), []).append(sl)
    curr_by_cpt: dict[str, list[dict]] = {}
    for sl in curr_slines:
        curr_by_cpt.setdefault(sl.get("cpt_code", ""), []).append(sl)

    for cpt in sorted(prev_cpts & curr_cpts):
        p_lines = prev_by_cpt.get(cpt, [])
        c_lines = curr_by_cpt.get(cpt, [])
        for i in range(min(len(p_lines), len(c_lines))):
            p_units = p_lines[i].get("units", 0)
            c_units = c_lines[i].get("units", 0)
            if p_units != c_units:
                categorized["units"].append(f"Changed units for {cpt} from {p_units} to {c_units}")

            p_chg = float(p_lines[i].get("charge_amount", 0.0))
            c_chg = float(c_lines[i].get("charge_amount", 0.0))
            if abs(p_chg - c_chg) > 0.01:
                categorized["charge"].append(f"Changed charge for {cpt} from {p_chg} to {c_chg}")

            p_date = p_lines[i].get("service_date", "")
            c_date = c_lines[i].get("service_date", "")
            if p_date != c_date and (p_date or c_date):
                categorized["service_date"].append(f"Changed service date for {cpt} from {p_date} to {c_date}")

    # Build flat summary
    all_changes: list[str] = []
    for cat in sorted(categorized):
        all_changes.extend(categorized[cat])
    result = dict(categorized)
    result["_flat"] = "; ".join(all_changes)
    return result


# ---------------------------------------------------------------------------
# create_or_update_lifecycle
# ---------------------------------------------------------------------------

async def create_or_update_lifecycle(
    session: AsyncSession,
    claim,
    prediction,
    validation_issues: list[dict],
    fixes_recommended: list[str],
) -> None:
    claim_id = _claim_field(claim, "claim_id")
    if not claim_id:
        return

    now = datetime.utcnow()
    freq_code = _claim_field(claim, "frequency_code", "1")
    payer_name = _claim_field(claim, "payer_name", "")
    patient_name = _patient_name(claim)

    features = {
        "total_charge": float(_claim_field(claim, "total_charge", 0.0)),
        "diagnosis_codes": list(_claim_field(claim, "diagnosis_codes", [])),
        "prior_auth_number": _claim_field(claim, "prior_auth_number", ""),
        "place_of_service": _claim_field(claim, "place_of_service", ""),
        "frequency_code": freq_code,
        "subscriber_id": _claim_field(claim, "subscriber_id", ""),
        "billing_provider_npi": _claim_field(claim, "billing_provider_npi", ""),
        "rendering_provider_npi": _claim_field(claim, "rendering_provider_npi", ""),
        "provider_taxonomy": _claim_field(claim, "provider_taxonomy", ""),
        "payer_id": _claim_field(claim, "payer_id", ""),
        "payer_name": _claim_field(claim, "payer_name", ""),
        "group_number": _claim_field(claim, "group_number", ""),
        "patient_dob": _claim_field(claim, "patient_dob", ""),
        "patient_gender": _claim_field(claim, "patient_gender", ""),
    }

    service_lines = _service_lines_summary(claim)

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

    existing = await lifecycle_repo.find_lifecycle(session, claim_id)

    if existing:
        # Resubmission: append new attempt
        attempts = existing.get("attempts", [])
        prev_attempt = attempts[-1] if attempts else {}
        fix_changes = detect_fix_applied(claim, prev_attempt) if prev_attempt else {}
        fix_applied = fix_changes.get("_flat", "") if fix_changes else ""
        attempt_number = existing.get("total_attempts", len(attempts)) + 1

        snapshot = _serialize_claim_snapshot(claim)
        new_attempt = {
            "attempt_number": attempt_number,
            "attempt_type": _classify_attempt_type(freq_code, attempt_number),
            "submitted_at": now,
            "frequency_code": freq_code,
            "fix_applied": fix_applied,
            "features": {**features, "_raw_snapshot": snapshot, "_fix_changes": fix_changes},
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

        await lifecycle_repo.add_attempt(session, claim_id, new_attempt, {
            "total_attempts": attempt_number,
            "current_status": "PENDING",
            "updated_at": now,
            "payer_name": payer_name,
            "patient_name": patient_name,
        })

        logger.info("Lifecycle updated — resubmission",
                    claim_id=claim_id, attempt=attempt_number,
                    fix_applied=fix_applied or "(none detected)")
    else:
        # First submission
        snapshot = _serialize_claim_snapshot(claim)
        first_attempt = {
            "attempt_number": 1,
            "attempt_type": _classify_attempt_type(freq_code, 1),
            "submitted_at": now,
            "frequency_code": freq_code,
            "features": {**features, "_raw_snapshot": snapshot},
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

        await lifecycle_repo.create_lifecycle(session, {
            "claim_id": claim_id,
            "payer_name": payer_name,
            "patient_name": patient_name,
            "current_status": "PENDING",
            "created_at": now,
            "updated_at": now,
        }, first_attempt)

        logger.info("Lifecycle created", claim_id=claim_id)


# ---------------------------------------------------------------------------
# update_lifecycle_outcome
# ---------------------------------------------------------------------------

async def update_lifecycle_outcome(
    session: AsyncSession,
    claim_id: str,
    remit,
    claim_doc: dict | None = None,
) -> dict | None:
    lifecycle = await lifecycle_repo.find_lifecycle(session, claim_id)

    if not lifecycle:
        # Backfill
        if not claim_doc:
            claim_doc = await claim_repo.find_claim(session, claim_id)
        if not claim_doc:
            return None

        now = datetime.utcnow()
        payer_name = claim_doc.get("payer_name", "")
        last = claim_doc.get("patient_last_name", "")
        first = claim_doc.get("patient_first_name", "")
        patient_name_str = ", ".join(p for p in (last, first) if p) or "UNKNOWN"

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

        slines = claim_doc.get("service_lines", [])
        features = {
            "total_charge": float(claim_doc.get("total_charge", 0.0)),
            "diagnosis_codes": list(claim_doc.get("diagnosis_codes", [])),
            "prior_auth_number": claim_doc.get("prior_auth_number", ""),
            "place_of_service": claim_doc.get("place_of_service", ""),
            "frequency_code": claim_doc.get("frequency_code", "1"),
            "subscriber_id": claim_doc.get("subscriber_id", ""),
            "billing_provider_npi": claim_doc.get("billing_provider_npi", ""),
            "rendering_provider_npi": claim_doc.get("rendering_provider_npi", ""),
            "provider_taxonomy": claim_doc.get("provider_taxonomy", ""),
            "payer_id": claim_doc.get("payer_id", ""),
            "payer_name": claim_doc.get("payer_name", ""),
            "group_number": claim_doc.get("group_number", ""),
            "patient_dob": claim_doc.get("patient_dob", ""),
            "patient_gender": claim_doc.get("patient_gender", ""),
        }

        pred = await prediction_repo.find_prediction(session, claim_id)
        risk_score = pred.get("risk_score", 0.0) if pred else 0.0
        risk_level = pred.get("risk_level", "LOW") if pred else "LOW"
        model_version = pred.get("model_version", "unknown") if pred else "unknown"

        backfill_freq = claim_doc.get("frequency_code", "1")
        backfilled_attempt = {
            "attempt_number": 1,
            "attempt_type": _classify_attempt_type(backfill_freq, 1),
            "submitted_at": claim_doc.get("created_at", now),
            "frequency_code": backfill_freq,
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

        await lifecycle_repo.create_lifecycle(session, {
            "claim_id": claim_id,
            "payer_name": payer_name,
            "patient_name": patient_name_str,
            "current_status": status,
            "created_at": claim_doc.get("created_at", now),
            "updated_at": now,
        }, backfilled_attempt)

        logger.info("Lifecycle backfilled from 835", claim_id=claim_id, status=status)
        return await lifecycle_repo.find_lifecycle(session, claim_id)

    # Determine status
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

    # Update the latest PENDING attempt
    attempt_num = await lifecycle_repo.update_attempt_outcome(
        session,
        lifecycle["id"],
        status=status,
        denial_codes=denial_codes,
        paid_amount=paid_amount,
        billed_amount=billed_amount,
    )

    if attempt_num is not None:
        await lifecycle_repo.update_lifecycle_status(session, claim_id, status)
        logger.info("Lifecycle outcome updated", claim_id=claim_id, attempt=attempt_num, status=status)

    return await lifecycle_repo.find_lifecycle(session, claim_id)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

async def get_lifecycle(session: AsyncSession, claim_id: str) -> dict | None:
    return await lifecycle_repo.get_lifecycle_detail(session, claim_id)


async def get_lifecycles(
    session: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    status: str | None = None,
    min_attempts: int | None = None,
    payer_name: str | None = None,
) -> tuple[list[dict], int]:
    return await lifecycle_repo.get_lifecycles_paginated(
        session,
        skip=skip, limit=limit,
        status=status, min_attempts=min_attempts, payer_name=payer_name,
    )


async def get_lifecycle_stats(session: AsyncSession) -> dict:
    return await lifecycle_repo.get_stats(session)
