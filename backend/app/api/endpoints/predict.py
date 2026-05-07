from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.services.prediction_service import predict_claim, predict_batch
from app.services.claim_rules import validate_claim
from app.services.decision_engine import decide
from app.schemas.claim import ParsedClaim
from app.schemas.prediction import PredictResponse, PredictBatchResponse
from app.repositories import claim_repo, prediction_repo

router = APIRouter(prefix="/predict", tags=["predict"])


class BatchPredictRequest(BaseModel):
    claim_ids: list[str] | None = None


@router.post("/{claim_id}", response_model=PredictResponse)
async def predict_single(claim_id: str, session: AsyncSession = Depends(get_db)):
    result = await predict_claim(session, claim_id)
    if not result:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim_doc = await claim_repo.find_claim(session, claim_id)
    if claim_doc:
        # Remove non-ParsedClaim fields
        for key in ["id", "validation_issues", "issue_count", "action", "action_label"]:
            claim_doc.pop(key, None)
        claim = ParsedClaim(**claim_doc)

        v = await validate_claim(claim, result)
        payer_name = claim.payer_name or ""
        primary_cpt = claim.service_lines[0].cpt_code if claim.service_lines else ""
        issue_dicts = [{"reason": iss.reason} for iss in v.issues]
        decision = decide(result.risk_score, len(v.issues), payer_name, primary_cpt, issues=issue_dicts)

        result.action = decision.action
        result.action_label = decision.action_label

        if decision.score_breakdown.final_score >= 0.7:
            risk_level = "HIGH"
        elif decision.score_breakdown.final_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        await prediction_repo.update_prediction_fields(session, claim_id, {
            "action": decision.action,
            "action_label": decision.action_label,
            "risk_score": decision.score_breakdown.final_score,
            "risk_level": risk_level,
        })

        await claim_repo.update_claim_fields(session, claim_id, {
            "validation_issues": [{"reason": iss.reason, "fixes": iss.fixes} for iss in v.issues],
            "issue_count": len(v.issues),
            "action": decision.action,
            "action_label": decision.action_label,
        })

        await session.commit()

    return result


@router.post("-batch", response_model=PredictBatchResponse)
async def predict_batch_endpoint(request: BatchPredictRequest, session: AsyncSession = Depends(get_db)):
    results = await predict_batch(session, request.claim_ids)
    await session.commit()
    return PredictBatchResponse(
        predicted_count=len(results),
        results=results,
    )
