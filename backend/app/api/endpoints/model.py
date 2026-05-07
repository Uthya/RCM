"""Model management endpoints — retrain, status, training history, versioning."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from app.api.deps import get_db
from app.services.model_trainer import (
    retrain_model, get_training_status, promote_model, rollback_model,
    DEFAULT_TRAINING_WINDOW_DAYS,
)
from app.services.remittance_service import CARC_DESCRIPTIONS
from app.repositories import training_repo, model_repo, outcome_repo, prediction_repo

logger = structlog.get_logger()
router = APIRouter(prefix="/model", tags=["model"])


@router.get("/training-status")
async def training_status(session: AsyncSession = Depends(get_db)):
    return await get_training_status(session)


@router.post("/retrain")
async def retrain(
    training_window_days: int = Query(
        default=DEFAULT_TRAINING_WINDOW_DAYS,
        ge=7,
        description="Only use training records from the last N days.",
    ),
    session: AsyncSession = Depends(get_db),
):
    result = await retrain_model(session, training_window_days=training_window_days)
    return result


@router.get("/training-data")
async def training_data_stats(session: AsyncSession = Depends(get_db)):
    total = await training_repo.count_records(session)
    denied = await training_repo.count_records(session, label=1)
    paid = total - denied

    top_codes_raw = await training_repo.get_top_denial_codes(session)
    top_denial_codes = [
        {
            "code": doc["code"],
            "description": CARC_DESCRIPTIONS.get(doc["code"], "Unknown"),
            "count": doc["count"],
        }
        for doc in top_codes_raw
    ]

    matched_claims = await outcome_repo.count_outcomes(session, status_filter=["paid", "denied"])
    gap = matched_claims - total

    version_dist = await training_repo.get_version_distribution(session)

    return {
        "total": total,
        "denied": denied,
        "paid": paid,
        "denial_rate": round(denied / max(total, 1), 4),
        "top_denial_codes": top_denial_codes,
        "gap_analysis": {
            "matched_claims": matched_claims,
            "training_records": total,
            "gap": gap,
            "message": f"{gap} matched claims missing training records — run backfill" if gap > 0
                       else "All matched claims have training records",
        },
        "model_version_distribution": [
            {"version": doc["version"], "count": doc["count"]}
            for doc in version_dist
        ],
    }


@router.get("/versions")
async def model_versions(session: AsyncSession = Depends(get_db)):
    versions = await model_repo.list_versions(session)
    return {
        "versions": [
            {
                "version": doc.get("version_str", f"v{doc['version']}"),
                "version_number": doc["version"],
                "trained_at": doc["trained_at"].isoformat() if doc.get("trained_at") else None,
                "is_active": doc.get("is_active", False),
                "status": doc.get("status", "active" if doc.get("is_active") else "unknown"),
                "promoted_at": doc["promoted_at"].isoformat() if doc.get("promoted_at") else None,
                "real_samples": doc.get("real_samples"),
                "synthetic_samples": doc.get("synthetic_samples"),
                "metrics": doc.get("metrics"),
                "top_denial_codes": doc.get("top_denial_codes", []),
            }
            for doc in versions
        ],
        "total_versions": len(versions),
    }


@router.post("/backfill-training-data")
async def backfill_training_data(session: AsyncSession = Depends(get_db)):
    """One-time migration: create ml_training_data records for previously matched claims."""
    existing_ids = await training_repo.get_all_claim_ids(session)

    from sqlalchemy import select
    from app.db.models import ClaimOutcome
    result = await session.execute(
        select(ClaimOutcome).where(ClaimOutcome.outcome_status.in_(["paid", "denied"]))
    )
    outcomes = result.scalars().all()

    created = 0
    skipped_no_prediction = 0
    skipped_existing = 0

    from app.core.feature_engineer import FEATURE_VERSION, FEATURE_COUNT, FEATURE_HASH

    for outcome_obj in outcomes:
        claim_id = outcome_obj.claim_id
        if claim_id in existing_ids:
            skipped_existing += 1
            continue

        prediction = await prediction_repo.find_prediction(session, claim_id)
        if not prediction or not prediction.get("features"):
            skipped_no_prediction += 1
            continue

        actual_outcome = outcome_obj.outcome_status or ""
        label = 1 if actual_outcome == "denied" else 0
        carc_codes = outcome_obj.carc_codes or []
        first_carc = carc_codes[0] if carc_codes else None

        training_doc = {
            "claim_id": claim_id,
            "attempt_number": 1,
            "is_first_attempt": True,
            "features": prediction["features"],
            "label": label,
            "actual_outcome": actual_outcome,
            "denial_code": first_carc,
            "denial_code_description": CARC_DESCRIPTIONS.get(first_carc, "Unknown") if first_carc else None,
            "all_carc_codes": carc_codes,
            "paid_amount": outcome_obj.paid_amount or 0.0,
            "billed_amount": prediction["features"].get("total_charge", 0.0),
            "model_version_at_prediction": prediction.get("model_version", "unknown"),
            "prediction_risk_score": prediction.get("risk_score"),
            "feature_version": prediction.get("feature_version", FEATURE_VERSION),
            "feature_count": FEATURE_COUNT,
            "feature_hash": FEATURE_HASH,
            "created_at": datetime.utcnow(),
        }

        await training_repo.upsert_training_record(session, training_doc)
        created += 1

    await session.commit()
    total_training = await training_repo.count_records(session)

    return {
        "status": "success",
        "created": created,
        "skipped_existing": skipped_existing,
        "skipped_no_prediction": skipped_no_prediction,
        "total_training_records": total_training,
    }


@router.post("/promote/{version}")
async def promote(
    version: int,
    force: bool = Query(default=False, description="Force promotion even if AUC drops below threshold"),
    session: AsyncSession = Depends(get_db),
):
    result = await promote_model(session, version, force=force)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@router.post("/rollback")
async def rollback(session: AsyncSession = Depends(get_db)):
    result = await rollback_model(session)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@router.get("/shadow-comparison")
async def shadow_comparison(session: AsyncSession = Depends(get_db)):
    from app.core.predictor import is_shadow_loaded, get_shadow_version, get_model_version

    if not is_shadow_loaded():
        return {
            "status": "no_shadow",
            "message": "No shadow model loaded. Retrain to create a candidate.",
        }

    shadow_docs = await prediction_repo.get_shadows(session)

    if not shadow_docs:
        return {
            "status": "no_data",
            "message": "Shadow model is loaded but no shadow predictions recorded yet.",
            "active_version": get_model_version(),
            "shadow_version": get_shadow_version(),
        }

    active_scores = [d["active_score"] for d in shadow_docs]
    shadow_scores = [d["shadow_score"] for d in shadow_docs]

    import numpy as np
    active_arr = np.array(active_scores)
    shadow_arr = np.array(shadow_scores)

    active_high = (active_arr >= 0.5).sum()
    shadow_high = (shadow_arr >= 0.5).sum()
    score_diff = float(np.mean(shadow_arr - active_arr))
    correlation = float(np.corrcoef(active_arr, shadow_arr)[0, 1]) if len(active_arr) > 1 else None

    docs_with_outcome = await prediction_repo.get_shadows_with_outcomes(session)

    auc_comparison = None
    if len(docs_with_outcome) > 10:
        from sklearn.metrics import roc_auc_score
        labels = [1 if d["actual_outcome"] == "denied" else 0 for d in docs_with_outcome]
        if len(set(labels)) > 1:
            active_auc = float(roc_auc_score(labels, [d["active_score"] for d in docs_with_outcome]))
            shadow_auc = float(roc_auc_score(labels, [d["shadow_score"] for d in docs_with_outcome]))
            auc_comparison = {
                "active_auc": round(active_auc, 4),
                "shadow_auc": round(shadow_auc, 4),
                "auc_delta": round(shadow_auc - active_auc, 4),
                "samples_with_outcome": len(docs_with_outcome),
            }

    active_model = await model_repo.get_active(session)
    shadow_version = get_shadow_version()
    shadow_version_num = int(shadow_version.lstrip("v")) if shadow_version.startswith("v") else None
    shadow_model_reg = await model_repo.find_by_version(session, shadow_version_num) if shadow_version_num else None

    return {
        "status": "ok",
        "active_version": get_model_version(),
        "shadow_version": get_shadow_version(),
        "total_shadow_predictions": len(shadow_docs),
        "score_comparison": {
            "mean_score_delta": round(score_diff, 4),
            "active_high_risk_count": int(active_high),
            "shadow_high_risk_count": int(shadow_high),
            "correlation": round(correlation, 4) if correlation is not None else None,
        },
        "auc_comparison": auc_comparison,
        "training_metrics": {
            "active": active_model.get("metrics") if active_model else None,
            "shadow": shadow_model_reg.get("metrics") if shadow_model_reg else None,
        },
    }
