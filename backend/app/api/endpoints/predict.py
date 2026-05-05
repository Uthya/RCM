from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.mongodb import get_db
from app.services.prediction_service import predict_claim, predict_batch
from app.services.claim_rules import validate_claim
from app.services.decision_engine import decide
from app.schemas.claim import ParsedClaim
from app.schemas.prediction import PredictResponse, PredictBatchResponse

router = APIRouter(prefix="/predict", tags=["predict"])


class BatchPredictRequest(BaseModel):
    claim_ids: list[str] | None = None  # None = predict all unpredicted


@router.post("/{claim_id}", response_model=PredictResponse)
async def predict_single(claim_id: str):
    result = await predict_claim(claim_id)
    if not result:
        raise HTTPException(status_code=404, detail="Claim not found")

    # Load claim for validation + decision
    db = get_db()
    claim_doc = await db.claims.find_one({"claim_id": claim_id})
    if claim_doc:
        claim_doc.pop("_id", None)
        claim = ParsedClaim(**claim_doc)

        # Run validation then decision with real issues
        v = await validate_claim(claim, result)
        payer_name = claim.payer_name or ""
        primary_cpt = claim.service_lines[0].cpt_code if claim.service_lines else ""
        issue_dicts = [{"reason": iss.reason} for iss in v.issues]
        decision = decide(result.risk_score, len(v.issues), payer_name, primary_cpt, issues=issue_dicts)

        # Update response with correct decision
        result.action = decision.action
        result.action_label = decision.action_label

        # Compute risk level from composite score
        if decision.score_breakdown.final_score >= 0.7:
            risk_level = "HIGH"
        elif decision.score_breakdown.final_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Persist correct decision to prediction doc
        await db.predictions.update_one(
            {"claim_id": claim_id},
            {"$set": {
                "action": decision.action,
                "action_label": decision.action_label,
                "risk_score": decision.score_breakdown.final_score,
                "risk_level": risk_level,
            }},
        )

        # Persist validation issues on claim doc
        await db.claims.update_one(
            {"claim_id": claim_id},
            {"$set": {
                "validation_issues": [{"reason": iss.reason, "fixes": iss.fixes} for iss in v.issues],
                "issue_count": len(v.issues),
                "action": decision.action,
                "action_label": decision.action_label,
            }},
        )

    return result


@router.post("-batch", response_model=PredictBatchResponse)
async def predict_batch_endpoint(request: BatchPredictRequest):
    results = await predict_batch(request.claim_ids)
    return PredictBatchResponse(
        predicted_count=len(results),
        results=results,
    )
