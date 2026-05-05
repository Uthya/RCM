from datetime import datetime
from pydantic import BaseModel, Field


class ServiceLinePayment(BaseModel):
    cpt_code: str = ""
    billed_amount: float = 0.0
    paid_amount: float = 0.0
    allowed_amount: float = 0.0
    adjustments: list[dict] = []  # [{group_code, reason_code, amount}]


class ParsedRemittance(BaseModel):
    claim_id: str  # CLP PCN / patient control number
    payer_control_number: str = ""
    claim_status_code: str = ""  # 1=paid, 4=denied, 19=denied, 22=partial
    claim_status: str = ""  # human-readable: paid / denied / partial

    billed_amount: float = 0.0
    paid_amount: float = 0.0
    patient_responsibility: float = 0.0

    # Payer / Payee
    payer_name: str = ""
    payee_name: str = ""
    payee_npi: str = ""

    # Payment info
    total_payment_amount: float = 0.0
    payment_method: str = ""
    payment_date: str = ""
    trace_number: str = ""

    # Claim-level adjustments
    adjustments: list[dict] = []  # [{group_code, reason_code, amount}]
    carc_codes: list[str] = []
    rarc_codes: list[str] = []

    # Service line payments
    service_lines: list[ServiceLinePayment] = []

    created_at: datetime = Field(default_factory=datetime.utcnow)


class RemittanceResponse(BaseModel):
    id: str = ""
    claim_id: str
    payer_control_number: str = ""
    claim_status: str = ""
    billed_amount: float = 0.0
    paid_amount: float = 0.0
    patient_responsibility: float = 0.0
    payer_name: str = ""
    payee_name: str = ""
    trace_number: str = ""
    payment_date: str = ""
    carc_codes: list[str] = []
    rarc_codes: list[str] = []
    adjustments: list[dict] = []
    service_lines: list[ServiceLinePayment] = []
    created_at: datetime | None = None


class RemittanceListResponse(BaseModel):
    remittances: list[RemittanceResponse]
    total: int
    skip: int
    limit: int
