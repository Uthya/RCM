import asyncio  # noqa
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException
import structlog

from app.db.mongodb import get_db
from app.parsers.parser_837 import parse_837
from app.parsers.parser_835 import parse_835
from app.services.claim_service import store_claims
from app.services.remittance_service import store_remittances
from app.services.prediction_service import predict_batch, BACKGROUND_THRESHOLD
from app.services.claim_rules import validate_claim, validation_to_dict
from app.services.claim_validator import validate_parsed_claims
from app.services.decision_engine import decide, decision_to_dict

logger = structlog.get_logger()
router = APIRouter(prefix="/upload", tags=["upload"])

# ── Background job storage (in-memory for now) ──
_background_jobs: dict[str, dict] = {}


@router.post("/837")
async def upload_837(file: UploadFile = File(...)):
    """Upload and parse an 837 Professional EDI file."""
    content = await file.read()
    raw = content.decode("utf-8", errors="replace")

    # Validate - check for ISA header
    if "ISA" not in raw[:100]:
        raise HTTPException(status_code=400, detail="Invalid EDI file: missing ISA header")

    # Parse
    try:
        claims = parse_837(raw)
    except Exception as e:
        logger.error("837 parse error", error=str(e))
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

    if not claims:
        raise HTTPException(status_code=400, detail="No claims found in file")

    # Pre-storage data validation
    dq = validate_parsed_claims(claims)
    if dq.rejected_claims:
        logger.warning("Rejected claims during pre-storage validation",
                        count=len(dq.rejected_claims))

    valid_claims = dq.valid_claims
    if not valid_claims:
        raise HTTPException(
            status_code=400,
            detail="All claims rejected during validation",
        )

    # Store only valid claims
    inserted = await store_claims(valid_claims)

    # Track upload
    db = get_db()
    await db.upload_history.insert_one({
        "filename": file.filename,
        "file_type": "837",
        "claim_count": len(valid_claims),
        "uploaded_at": datetime.utcnow(),
    })

    claim_ids = [c.claim_id for c in valid_claims]

    # Build data quality stats for response
    data_quality = {
        "total_parsed": len(claims),
        "valid": len(dq.valid_claims),
        "rejected": len(dq.rejected_claims),
        "warnings": len(dq.warnings),
        "duplicates": len(dq.duplicate_ids),
        "rejected_details": dq.rejected_claims[:10],
    }

    # For large files, run prediction in background
    if len(valid_claims) > BACKGROUND_THRESHOLD:
        job_id = str(uuid4())
        _background_jobs[job_id] = {"status": "processing", "results": None}
        asyncio.create_task(_predict_and_validate_background(job_id, claim_ids, valid_claims))
        return {
            "message": f"Parsed {len(claims)} claims from 837 file. Predictions running in background.",
            "claims_parsed": len(claims),
            "claims_stored": inserted,
            "predictions_made": 0,
            "job_id": job_id,
            "status": "processing",
            "claim_ids": claim_ids,
            "data_quality": data_quality,
        }

    # Auto-predict
    predictions = await predict_batch(claim_ids)

    # Build response and save validation issues to DB
    response = await _build_837_response(valid_claims, predictions, inserted, db)

    # Persist validation issues on claim docs
    for cv in response["risk_summary"]["claim_errors"]:
        await db.claims.update_one(
            {"claim_id": cv["claim_id"]},
            {"$set": {
                "validation_issues": cv["issues"],
                "issue_count": len(cv["issues"]),
                "action": cv.get("action", ""),
                "action_label": cv.get("action_label", ""),
            }},
        )

    # ── Lifecycle tracking ──
    try:
        from app.services.lifecycle_service import create_or_update_lifecycle
        flagged_lookup = {
            cv["claim_id"]: cv for cv in response["risk_summary"]["claim_errors"]
        }
        pred_lookup_lc = {p.claim_id: p for p in predictions}
        for c in valid_claims:
            cv_data = flagged_lookup.get(c.claim_id)
            issues = cv_data["issues"] if cv_data else []
            fixes = []
            for iss in issues:
                fixes.extend(iss.get("fixes", []))
            await create_or_update_lifecycle(
                claim=c,
                prediction=pred_lookup_lc.get(c.claim_id),
                validation_issues=issues,
                fixes_recommended=fixes,
            )
    except Exception as e:
        logger.warning("Lifecycle tracking failed", error=str(e))

    response["data_quality"] = data_quality
    return response


async def _predict_and_validate_background(job_id: str, claim_ids: list[str], claims):
    """Background task for large file prediction."""
    try:
        db = get_db()
        predictions = await predict_batch(claim_ids)
        result = await _build_837_response(claims, predictions, len(claims), db)

        # Persist validation issues on claim docs (same as foreground path)
        for cv in result["risk_summary"]["claim_errors"]:
            await db.claims.update_one(
                {"claim_id": cv["claim_id"]},
                {"$set": {
                    "validation_issues": cv["issues"],
                    "issue_count": len(cv["issues"]),
                    "action": cv.get("action", ""),
                    "action_label": cv.get("action_label", ""),
                }},
            )

        # ── Lifecycle tracking (background path) ──
        try:
            from app.services.lifecycle_service import create_or_update_lifecycle
            flagged_lookup = {
                cv["claim_id"]: cv for cv in result["risk_summary"]["claim_errors"]
            }
            pred_lookup_lc = {p.claim_id: p for p in predictions}
            for c in claims:
                cv_data = flagged_lookup.get(c.claim_id)
                issues = cv_data["issues"] if cv_data else []
                fixes = []
                for iss in issues:
                    fixes.extend(iss.get("fixes", []))
                await create_or_update_lifecycle(
                    claim=c,
                    prediction=pred_lookup_lc.get(c.claim_id),
                    validation_issues=issues,
                    fixes_recommended=fixes,
                )
        except Exception as e:
            logger.warning("Lifecycle tracking failed (background)", error=str(e))

        _background_jobs[job_id] = {"status": "completed", "results": result}
    except Exception as e:
        logger.error("Background prediction failed", job_id=job_id, error=str(e))
        _background_jobs[job_id] = {"status": "failed", "error": str(e)}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll for background job results."""
    job = _background_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "completed":
        return job["results"]
    return {"status": job["status"], "job_id": job_id}


async def _build_837_response(claims, predictions, inserted, db) -> dict:
    """Build the per-claim output response for 837 upload."""
    # Payer breakdown
    payer_counts: dict[str, int] = {}
    for c in claims:
        payer_counts[c.payer_name or "Unknown"] = payer_counts.get(c.payer_name or "Unknown", 0) + 1

    # Build prediction lookup
    pred_lookup = {p.claim_id: p for p in predictions}

    # ── Compute patient-level concentration risk ──
    # Detect patients with unusually high claim volume (audit risk)
    patient_claim_counts: dict[str, int] = {}
    for c in claims:
        patient_key = f"{c.patient_first_name} {c.patient_last_name}".strip().upper()
        if patient_key and patient_key != "UNKNOWN":
            patient_claim_counts[patient_key] = patient_claim_counts.get(patient_key, 0) + 1

    # Flag patients with 5+ claims in a single upload as concentration risk
    CONCENTRATION_THRESHOLD = 5
    high_volume_patients: set[str] = {
        p for p, cnt in patient_claim_counts.items() if cnt >= CONCENTRATION_THRESHOLD
    }

    # ── Per-claim validation with decision engine ──
    claim_lookup = {c.claim_id: c for c in claims}
    claim_results: list[dict] = []

    for p in predictions:
        c = claim_lookup.get(p.claim_id)
        if not c:
            continue

        # Validate claim
        v = await validate_claim(c, p)

        # Attach concentration risk if applicable
        patient_key = f"{c.patient_first_name} {c.patient_last_name}".strip().upper()
        if patient_key in high_volume_patients:
            from app.services.claim_rules import ClaimIssue
            count = patient_claim_counts[patient_key]
            v.issues.append(ClaimIssue(
                reason=(
                    f"Patient concentration risk — high-volume billing pattern ({count} claims)\n"
                    f"Same patient billed {count} times in this submission, which may trigger "
                    f"payer audit flags for overutilization"
                ),
                fixes=[
                    "Review medical necessity documentation for each visit",
                    "Ensure each claim has distinct dates of service and clinical justification",
                    "Consider grouping services where clinically appropriate",
                ],
            ))

        # Decision engine with per-issue severity
        payer_name = c.payer_name or ""
        primary_cpt = c.service_lines[0].cpt_code if c.service_lines else ""
        issue_dicts = [{"reason": iss.reason} for iss in v.issues]
        decision = decide(p.risk_score, len(v.issues), payer_name, primary_cpt, issues=issue_dicts)

        # Attach decision to validation — use composite score as the display score
        v.action = decision.action
        v.action_label = decision.action_label
        v.score_breakdown = decision_to_dict(decision)["score_breakdown"]
        v.risk_score = decision.score_breakdown.final_score  # composite replaces raw ML

        # Update risk_level to match composite score
        if decision.score_breakdown.final_score >= 0.7:
            v.risk_level = "HIGH"
        elif decision.score_breakdown.final_score >= 0.3:
            v.risk_level = "MEDIUM"
        else:
            v.risk_level = "LOW"

        # Persist correct decision back to prediction doc (single source of truth)
        await db.predictions.update_one(
            {"claim_id": p.claim_id},
            {"$set": {
                "action": decision.action,
                "action_label": decision.action_label,
                "risk_score": decision.score_breakdown.final_score,
                "risk_level": v.risk_level,
            }},
        )

        vd = validation_to_dict(v)
        claim_results.append(vd)

    # Sort by composite final_score desc (not raw ML score)
    claim_results.sort(
        key=lambda x: x.get("score_breakdown", {}).get("final_score", x["risk_score"]),
        reverse=True,
    )

    # ── File-level summary derived FROM decision engine composite scores ──
    composite_scores = [
        cr.get("score_breakdown", {}).get("final_score", cr["risk_score"])
        for cr in claim_results
    ]
    auto_submit_count = sum(1 for c in claim_results if c["action"] == "auto_submit")
    review_count = sum(1 for c in claim_results if c["action"] == "review")
    fix_count = sum(1 for c in claim_results if c["action"] == "fix_required")
    avg_risk = round(sum(composite_scores) / len(composite_scores), 4) if composite_scores else 0.0
    max_risk = round(max(composite_scores), 4) if composite_scores else 0.0
    min_risk = round(min(composite_scores), 4) if composite_scores else 0.0

    if avg_risk < 0.3:
        file_risk_level = "LOW"
    elif avg_risk <= 0.7:
        file_risk_level = "MEDIUM"
    else:
        file_risk_level = "HIGH"

    # ── Aggregate file-level reasons & fixes ──
    reason_counts: dict[str, int] = {}
    fix_counts: dict[str, int] = {}
    for cv in claim_results:
        for iss in cv["issues"]:
            label = iss["reason"].split("\n")[0]
            reason_counts[label] = reason_counts.get(label, 0) + 1
            for fix in iss["fixes"]:
                fix_counts[fix] = fix_counts.get(fix, 0) + 1

    file_top_reasons = sorted(reason_counts.items(), key=lambda x: -x[1])[:5]
    file_top_fixes = sorted(fix_counts.items(), key=lambda x: -x[1])[:5]

    # Top risk claims (from claim_results which have composite scores)
    top_risk_claims = [
        {
            "claim_id": cr["claim_id"],
            "risk_score": cr["risk_score"],
            "risk_level": cr["risk_level"],
            "top_reason": cr["top_factors"][0]["name"] if cr["top_factors"] else "",
        }
        for cr in claim_results[:5]  # already sorted desc by composite score
    ]

    # Only include claims with issues or elevated composite risk in claim_errors
    flagged = [cv for cv in claim_results if cv["issues"] or cv["risk_score"] >= 0.3]

    return {
        "message": f"Parsed {len(claims)} claims from 837 file",
        "claims_parsed": len(claims),
        "claims_stored": inserted,
        "predictions_made": len(predictions),
        "payer_breakdown": payer_counts,
        "claim_ids": [c.claim_id for c in claims],
        "risk_summary": {
            "avg_risk_score": avg_risk,
            "max_risk_score": max_risk,
            "min_risk_score": min_risk,
            "file_risk_level": file_risk_level,
            "auto_submit_count": auto_submit_count,
            "review_count": review_count,
            "needs_fix_count": fix_count,
            "top_risk_claims": top_risk_claims,
            "claim_errors": flagged,
            "file_top_reasons": [{"reason": r, "count": c} for r, c in file_top_reasons],
            "file_top_fixes": [{"fix": f, "count": c} for f, c in file_top_fixes],
            "patient_concentration": {
                "total_patients": len(patient_claim_counts),
                "high_volume_patients": len(high_volume_patients),
                "threshold": CONCENTRATION_THRESHOLD,
                "top_patients": sorted(
                    [{"patient": p, "claim_count": cnt}
                     for p, cnt in patient_claim_counts.items()
                     if cnt >= CONCENTRATION_THRESHOLD],
                    key=lambda x: -x["claim_count"],
                )[:10],
            },
        },
    }


@router.post("/835")
async def upload_835(file: UploadFile = File(...)):
    """Upload and parse an 835 Remittance Advice EDI file."""
    content = await file.read()
    raw = content.decode("utf-8", errors="replace")

    # Validate
    if "ISA" not in raw[:100]:
        raise HTTPException(status_code=400, detail="Invalid EDI file: missing ISA header")

    # Parse
    try:
        remittances = parse_835(raw)
    except Exception as e:
        logger.error("835 parse error", error=str(e))
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")

    if not remittances:
        raise HTTPException(status_code=400, detail="No remittance records found in file")

    # Store and match
    summary = await store_remittances(remittances)

    # Track upload
    db = get_db()
    await db.upload_history.insert_one({
        "filename": file.filename,
        "file_type": "835",
        "claim_count": len(remittances),
        "uploaded_at": datetime.utcnow(),
    })

    # ── Check auto-retrain trigger ──
    from app.services.model_trainer import (
        AUTO_RETRAIN_THRESHOLD,
        AUTO_RETRAIN_INTERVAL_DAYS,
        validate_training_data,
        retrain_model,
    )
    from datetime import timedelta

    auto_retrain_triggered = False
    training_status = None

    total_training = await db.ml_training_data.count_documents({})

    if total_training >= AUTO_RETRAIN_THRESHOLD:
        last = await db.training_history.find_one(sort=[("trained_at", -1)])
        cooldown_ok = (
            not last
            or (datetime.utcnow() - last["trained_at"]) > timedelta(days=AUTO_RETRAIN_INTERVAL_DAYS)
        )
        if cooldown_ok:
            quality = await validate_training_data(db)
            if quality["passed"]:
                asyncio.create_task(retrain_model(db))
                auto_retrain_triggered = True
                training_status = "retraining_in_background"
                logger.info("Auto-retrain triggered", training_records=total_training)
            else:
                training_status = "retrain_skipped_quality"
                logger.warning("Auto-retrain skipped: quality check failed", issues=quality["issues"])
        else:
            training_status = "retrain_cooldown"
    else:
        records_until = AUTO_RETRAIN_THRESHOLD - total_training
        training_status = f"need_{records_until}_more_training_records"

    return {
        "message": f"Parsed {len(remittances)} remittance records from 835 file",
        "records_parsed": len(remittances),
        "records_stored": summary["inserted"],
        "matched_to_claims": summary["matched"],
        "denied_count": summary["denied"],
        "total_paid": summary["total_paid"],
        "training_records_created": summary.get("training_records_created", 0),
        "total_training_records": summary.get("total_training_records", 0),
        "training_status": training_status,
        "auto_retrain_triggered": auto_retrain_triggered,
        "total_matched_claims": summary.get("total_matched_claims", 0),
        "ready_to_retrain": total_training >= AUTO_RETRAIN_THRESHOLD,
        "records_until_retrain": max(0, AUTO_RETRAIN_THRESHOLD - total_training),
    }
