"""Orchestrator: features -> predict -> explain -> store predictions (decision deferred to caller)."""

import asyncio
import structlog

from app.config import settings
from app.db.mongodb import get_db
from app.core.feature_engineer import (
    compute_features_from_claim,
    enrich_with_historical_rates,
    features_to_dataframe,
)
from app.core.predictor import (
    predict_proba, get_model_version,
    predict_shadow, is_shadow_loaded, get_shadow_version,
)
from app.core.explainer import explain
from app.schemas.prediction import DenialPrediction, PredictResponse, RiskFactor

logger = structlog.get_logger()

from datetime import datetime

BATCH_CHUNK_SIZE = 100  # claims per prediction chunk
BACKGROUND_THRESHOLD = 500  # claims above this run in background


async def _log_shadow_prediction(
    db, claim_id: str, active_score: float, shadow_score: float, model_ver: str,
) -> None:
    """Log shadow model prediction alongside the active score for later comparison."""
    await db.shadow_predictions.update_one(
        {"claim_id": claim_id},
        {"$set": {
            "claim_id": claim_id,
            "active_score": active_score,
            "active_version": model_ver,
            "shadow_score": shadow_score,
            "shadow_version": get_shadow_version(),
            "scored_at": datetime.utcnow(),
        }},
        upsert=True,
    )


def _classify_risk(score: float) -> str:
    if score >= settings.RISK_HIGH_THRESHOLD:
        return "HIGH"
    elif score >= settings.RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


async def predict_claim(claim_id: str) -> PredictResponse | None:
    """Run prediction for a single claim."""
    db = get_db()

    claim_doc = await db.claims.find_one({"claim_id": claim_id})
    if not claim_doc:
        return None

    # Feature engineering
    features = compute_features_from_claim(claim_doc)
    features = await enrich_with_historical_rates(features, claim_doc)

    # Build DataFrame
    df = features_to_dataframe([features])

    # Predict
    proba = predict_proba(df)
    risk_score = round(float(proba[0]), 4)
    risk_level = _classify_risk(risk_score)

    # Explain
    explanations = explain(df, top_n=3)
    risk_factors = explanations[0] if explanations else []

    # Store prediction (decision deferred until after validation)
    prediction = DenialPrediction(
        claim_id=claim_id,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_factors=risk_factors,
        action="",
        action_label="",
        features=features,
        model_version=get_model_version(),
    )

    await db.predictions.update_one(
        {"claim_id": claim_id},
        {"$set": prediction.model_dump()},
        upsert=True,
    )

    logger.info("Prediction stored (pending decision)", claim_id=claim_id,
                risk_score=risk_score, risk_level=risk_level)

    # Shadow scoring: score with candidate model if loaded (non-blocking)
    if is_shadow_loaded():
        shadow_proba = predict_shadow(df)
        if shadow_proba is not None:
            await _log_shadow_prediction(
                db, claim_id, risk_score, round(float(shadow_proba[0]), 4),
                get_model_version(),
            )

    return PredictResponse(
        claim_id=claim_id,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_factors=risk_factors,
        action="",
        action_label="",
    )


async def predict_batch(claim_ids: list[str] | None = None) -> list[PredictResponse]:
    """Run predictions for multiple claims with chunked processing."""
    db = get_db()

    if claim_ids is None:
        # Find claims without predictions
        predicted_ids = await db.predictions.distinct("claim_id")
        query = {"claim_id": {"$nin": predicted_ids}} if predicted_ids else {}
        all_claims = await db.claims.find(query).to_list(10000)
        claim_ids = [c["claim_id"] for c in all_claims]
    else:
        all_claims = None  # will load per chunk

    if not claim_ids:
        return []

    # Split into chunks
    chunks = [claim_ids[i:i + BATCH_CHUNK_SIZE] for i in range(0, len(claim_ids), BATCH_CHUNK_SIZE)]
    all_results: list[PredictResponse] = []

    for chunk_ids in chunks:
        # Load claims for this chunk
        if all_claims is not None:
            claims = [c for c in all_claims if c["claim_id"] in set(chunk_ids)]
        else:
            claims = await db.claims.find({"claim_id": {"$in": chunk_ids}}).to_list(BATCH_CHUNK_SIZE)

        if not claims:
            continue

        # Parallel feature enrichment within chunk
        async def _enrich(doc):
            feats = compute_features_from_claim(doc)
            return await enrich_with_historical_rates(feats, doc)

        enrichment_tasks = [_enrich(doc) for doc in claims]
        feature_list = await asyncio.gather(*enrichment_tasks)

        # Build DataFrame and predict (vectorized)
        df = features_to_dataframe(feature_list)
        probas = predict_proba(df)
        explanations = explain(df, top_n=3)
        model_ver = get_model_version()

        # Shadow scoring for the whole chunk
        shadow_probas = predict_shadow(df) if is_shadow_loaded() else None

        # Store and collect results
        for i, claim_doc in enumerate(claims):
            risk_score = round(float(probas[i]), 4)
            risk_level = _classify_risk(risk_score)
            risk_factors = explanations[i] if i < len(explanations) else []

            # Store prediction (decision deferred until after validation)
            prediction = DenialPrediction(
                claim_id=claim_doc["claim_id"],
                risk_score=risk_score,
                risk_level=risk_level,
                risk_factors=risk_factors,
                action="",
                action_label="",
                features=feature_list[i],
                model_version=model_ver,
            )

            await db.predictions.update_one(
                {"claim_id": claim_doc["claim_id"]},
                {"$set": prediction.model_dump()},
                upsert=True,
            )

            # Log shadow prediction
            if shadow_probas is not None:
                await _log_shadow_prediction(
                    db, claim_doc["claim_id"], risk_score,
                    round(float(shadow_probas[i]), 4), model_ver,
                )

            all_results.append(PredictResponse(
                claim_id=claim_doc["claim_id"],
                risk_score=risk_score,
                risk_level=risk_level,
                risk_factors=risk_factors,
                action="",
                action_label="",
            ))

    logger.info("Batch prediction complete", count=len(all_results),
                chunks=len(chunks), chunk_size=BATCH_CHUNK_SIZE)
    return all_results
