"""Orchestrator: features -> predict -> explain -> store predictions (decision deferred to caller)."""

import asyncio
import structlog
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.feature_engineer import (
    FEATURE_VERSION,
    compute_features_from_claim,
    enrich_with_historical_rates,
    features_to_dataframe,
    align_features_for_model,
)
from app.core.predictor import (
    predict_proba, get_model_version,
    predict_shadow, is_shadow_loaded, get_shadow_version,
)
from app.core.explainer import explain
from app.schemas.prediction import DenialPrediction, PredictResponse, RiskFactor
from app.repositories import claim_repo, prediction_repo

logger = structlog.get_logger()

BATCH_CHUNK_SIZE = 100
BACKGROUND_THRESHOLD = 500


async def _log_shadow_prediction(
    session: AsyncSession, claim_id: str, active_score: float, shadow_score: float, model_ver: str,
) -> None:
    await prediction_repo.upsert_shadow(session, {
        "claim_id": claim_id,
        "attempt_number": 1,
        "active_score": active_score,
        "active_version": model_ver,
        "shadow_score": shadow_score,
        "shadow_version": get_shadow_version(),
        "scored_at": datetime.utcnow(),
    })


def _classify_risk(score: float) -> str:
    if score >= settings.RISK_HIGH_THRESHOLD:
        return "HIGH"
    elif score >= settings.RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


async def predict_claim(session: AsyncSession, claim_id: str) -> PredictResponse | None:
    """Run prediction for a single claim."""
    claim_doc = await claim_repo.find_claim(session, claim_id)
    if not claim_doc:
        return None

    features = compute_features_from_claim(claim_doc)
    features = await enrich_with_historical_rates(session, features, claim_doc)

    df = features_to_dataframe([features])
    df = align_features_for_model(df)
    proba = predict_proba(df)
    risk_score = round(float(proba[0]), 4)
    risk_level = _classify_risk(risk_score)

    explanations = explain(df, top_n=3)
    risk_factors = explanations[0] if explanations else []

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

    pred_doc = prediction.model_dump()
    pred_doc["feature_version"] = FEATURE_VERSION
    pred_doc["attempt_number"] = 1

    await prediction_repo.upsert_prediction(session, pred_doc)

    logger.info("Prediction stored (pending decision)", claim_id=claim_id,
                risk_score=risk_score, risk_level=risk_level)

    if is_shadow_loaded():
        shadow_proba = predict_shadow(df)
        if shadow_proba is not None:
            await _log_shadow_prediction(
                session, claim_id, risk_score, round(float(shadow_proba[0]), 4),
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


async def predict_batch(session: AsyncSession, claim_ids: list[str] | None = None) -> list[PredictResponse]:
    """Run predictions for multiple claims with chunked processing."""
    if claim_ids is None:
        predicted_ids = await prediction_repo.get_predicted_ids(session)
        all_claims = await claim_repo.get_claims_by_ids(session, []) if not predicted_ids else []
        if not predicted_ids:
            # Get all claims
            from sqlalchemy import select
            from app.db.models import Claim
            result = await session.execute(select(Claim))
            all_claim_objs = result.scalars().all()
            all_claims = [claim_repo._claim_to_dict(c) for c in all_claim_objs]
        else:
            unpredicted_ids = await prediction_repo.get_unpredicted_claim_ids(session)
            all_claims = await claim_repo.get_claims_by_ids(session, unpredicted_ids)
        claim_ids = [c["claim_id"] for c in all_claims]
    else:
        all_claims = None

    if not claim_ids:
        return []

    chunks = [claim_ids[i:i + BATCH_CHUNK_SIZE] for i in range(0, len(claim_ids), BATCH_CHUNK_SIZE)]
    all_results: list[PredictResponse] = []

    for chunk_ids in chunks:
        if all_claims is not None:
            claims = [c for c in all_claims if c["claim_id"] in set(chunk_ids)]
        else:
            claims = await claim_repo.get_claims_by_ids(session, chunk_ids)

        if not claims:
            continue

        async def _enrich(doc):
            feats = compute_features_from_claim(doc)
            return await enrich_with_historical_rates(session, feats, doc)

        enrichment_tasks = [_enrich(doc) for doc in claims]
        feature_list = await asyncio.gather(*enrichment_tasks)

        df = features_to_dataframe(feature_list)
        df = align_features_for_model(df)
        probas = predict_proba(df)
        explanations = explain(df, top_n=3)
        model_ver = get_model_version()

        shadow_probas = predict_shadow(df) if is_shadow_loaded() else None

        for i, claim_doc in enumerate(claims):
            risk_score = round(float(probas[i]), 4)
            risk_level = _classify_risk(risk_score)
            risk_factors = explanations[i] if i < len(explanations) else []

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

            pred_doc = prediction.model_dump()
            pred_doc["feature_version"] = FEATURE_VERSION
            pred_doc["attempt_number"] = 1

            await prediction_repo.upsert_prediction(session, pred_doc)

            if shadow_probas is not None:
                await _log_shadow_prediction(
                    session, claim_doc["claim_id"], risk_score,
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
