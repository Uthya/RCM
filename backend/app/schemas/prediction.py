from datetime import datetime
from pydantic import BaseModel, Field


class RiskFactor(BaseModel):
    feature: str
    display_name: str
    impact: float  # SHAP value
    value: str = ""  # feature value for context


class DenialPrediction(BaseModel):
    claim_id: str
    risk_score: float  # 0.0 - 1.0
    risk_level: str  # HIGH, MEDIUM, LOW
    risk_factors: list[RiskFactor] = []
    action: str = ""           # auto_submit | review | fix_required
    action_label: str = ""     # Auto Submit | Review | Fix Required
    features: dict = {}        # feature dict used for this prediction
    model_version: str = "unknown"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PredictResponse(BaseModel):
    claim_id: str
    risk_score: float
    risk_level: str
    risk_factors: list[RiskFactor] = []
    action: str = ""
    action_label: str = ""


class PredictBatchResponse(BaseModel):
    predicted_count: int
    results: list[PredictResponse]
