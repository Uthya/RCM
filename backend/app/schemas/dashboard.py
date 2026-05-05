from pydantic import BaseModel


class DashboardSummary(BaseModel):
    total_claims: int = 0
    total_predicted: int = 0
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    total_remittances: int = 0
    denial_rate: float = 0.0  # from 835 outcomes
    total_billed: float = 0.0
    total_paid: float = 0.0


class RiskBucket(BaseModel):
    range_label: str  # e.g. "0.0-0.1"
    count: int


class RiskDistribution(BaseModel):
    buckets: list[RiskBucket]


class PayerStat(BaseModel):
    payer_name: str
    payer_id: str
    total_claims: int
    denied_count: int
    denial_rate: float


class PayerStatsResponse(BaseModel):
    payers: list[PayerStat]
