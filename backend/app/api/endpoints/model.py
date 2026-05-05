"""Model management endpoints — retrain, status, training history, versioning."""

from datetime import datetime

from fastapi import APIRouter
import structlog

from app.db.mongodb import get_db
from app.services.model_trainer import retrain_model, get_training_status
from app.services.remittance_service import CARC_DESCRIPTIONS

logger = structlog.get_logger()
router = APIRouter(prefix="/model", tags=["model"])


@router.get("/training-status")
async def training_status():
    """Get current training data availability and last training info."""
    db = get_db()
    return await get_training_status(db)


@router.post("/retrain")
async def retrain():
    """Retrain the model using pre-joined training data from ml_training_data."""
    db = get_db()
    result = await retrain_model(db)
    return result


@router.get("/training-data")
async def training_data_stats():
    """Stats on ml_training_data: total, class distribution, top denial codes, gap analysis."""
    db = get_db()

    total = await db.ml_training_data.count_documents({})
    denied = await db.ml_training_data.count_documents({"label": 1})
    paid = total - denied

    # Top denial codes
    pipeline = [
        {"$match": {"label": 1, "denial_code": {"$ne": None}}},
        {"$group": {"_id": "$denial_code", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]
    top_codes_raw = await db.ml_training_data.aggregate(pipeline).to_list(10)
    top_denial_codes = [
        {
            "code": doc["_id"],
            "description": CARC_DESCRIPTIONS.get(doc["_id"], "Unknown"),
            "count": doc["count"],
        }
        for doc in top_codes_raw
    ]

    # Gap analysis: matched claims with predictions but no training record
    matched_claims = await db.claims.count_documents(
        {"actual_outcome": {"$in": ["paid", "denied"]}}
    )
    gap = matched_claims - total

    # Model version distribution in training data
    version_pipeline = [
        {"$group": {"_id": "$model_version_at_prediction", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    version_dist = await db.ml_training_data.aggregate(version_pipeline).to_list(20)

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
            {"version": doc["_id"], "count": doc["count"]}
            for doc in version_dist
        ],
    }


@router.get("/versions")
async def model_versions():
    """List all model versions from model_registry with metrics."""
    db = get_db()

    cursor = db.model_registry.find().sort("version", -1)
    versions = await cursor.to_list(100)

    return {
        "versions": [
            {
                "version": doc.get("version_str", f"v{doc['version']}"),
                "version_number": doc["version"],
                "trained_at": doc["trained_at"].isoformat() if doc.get("trained_at") else None,
                "is_active": doc.get("is_active", False),
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
async def backfill_training_data():
    """One-time migration: create ml_training_data records for previously matched claims.

    Finds claims with actual_outcome that have predictions with features
    but no ml_training_data record yet.
    """
    db = get_db()

    # Get all claim_ids already in ml_training_data
    existing_ids = set()
    cursor = db.ml_training_data.find({}, {"claim_id": 1})
    async for doc in cursor:
        existing_ids.add(doc["claim_id"])

    # Find matched claims not yet in training data
    claims_cursor = db.claims.find(
        {"actual_outcome": {"$in": ["paid", "denied"]}},
        {"claim_id": 1, "actual_outcome": 1, "paid_amount": 1,
         "carc_codes": 1, "carc_descriptions": 1},
    )
    claims = await claims_cursor.to_list(100_000)

    created = 0
    skipped_no_prediction = 0
    skipped_existing = 0

    for claim in claims:
        claim_id = claim["claim_id"]
        if claim_id in existing_ids:
            skipped_existing += 1
            continue

        # Look up prediction with features
        prediction = await db.predictions.find_one({"claim_id": claim_id})
        if not prediction or not prediction.get("features"):
            skipped_no_prediction += 1
            continue

        label = 1 if claim.get("actual_outcome") == "denied" else 0
        carc_codes = claim.get("carc_codes", [])
        first_carc = carc_codes[0] if carc_codes else None

        training_doc = {
            "claim_id": claim_id,
            "features": prediction["features"],
            "label": label,
            "actual_outcome": claim.get("actual_outcome"),
            "denial_code": first_carc,
            "denial_code_description": CARC_DESCRIPTIONS.get(first_carc, "Unknown") if first_carc else None,
            "all_carc_codes": carc_codes,
            "paid_amount": claim.get("paid_amount", 0.0),
            "billed_amount": prediction["features"].get("total_charge", 0.0),
            "model_version_at_prediction": prediction.get("model_version", "unknown"),
            "prediction_risk_score": prediction.get("risk_score"),
            "created_at": datetime.utcnow(),
        }

        await db.ml_training_data.update_one(
            {"claim_id": claim_id},
            {"$set": training_doc},
            upsert=True,
        )
        created += 1

    total_training = await db.ml_training_data.count_documents({})

    return {
        "status": "success",
        "created": created,
        "skipped_existing": skipped_existing,
        "skipped_no_prediction": skipped_no_prediction,
        "total_training_records": total_training,
    }
