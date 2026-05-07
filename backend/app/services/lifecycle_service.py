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


# ---------------------------------------------------------------------------
# detect_fix_applied
# ---------------------------------------------------------------------------

def detect_fix_applied(claim, previous_attempt: dict) -> str:
    changes: list[str] = []
    prev_features = previous_attempt.get("features", {})

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

    prev_dx = set(prev_features.get("diagnosis_codes", []))
    curr_dx = set(_claim_field(claim, "diagnosis_codes", []))
    added_dx = curr_dx - prev_dx
    removed_dx = prev_dx - curr_dx
    if added_dx:
        changes.append(f"Added diagnosis {', '.join(sorted(added_dx))}")
    if removed_dx:
        changes.append(f"Removed diagnosis {', '.join(sorted(removed_dx))}")

    prev_auth = prev_features.get("prior_auth_number", "")
    curr_auth = _claim_field(claim, "prior_auth_number", "")
    if curr_auth and not prev_auth:
        changes.append("Added prior authorization")
    elif prev_auth and not curr_auth:
        changes.append("Removed prior authorization")

    prev_charge = prev_features.get("total_charge", 0.0)
    curr_charge = float(_claim_field(claim, "total_charge", 0.0))
    if prev_charge and curr_charge and abs(curr_charge - prev_charge) > 0.01:
        changes.append(f"Changed total charge from {prev_charge} to {curr_charge}")

    return "; ".join(changes)


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
        fix_applied = detect_fix_applied(claim, prev_attempt) if prev_attempt else ""
        attempt_number = existing.get("total_attempts", len(attempts)) + 1

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
        }

        pred = await prediction_repo.find_prediction(session, claim_id)
        risk_score = pred.get("risk_score", 0.0) if pred else 0.0
        risk_level = pred.get("risk_level", "LOW") if pred else "LOW"
        model_version = pred.get("model_version", "unknown") if pred else "unknown"

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
