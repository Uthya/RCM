from datetime import date, datetime
from pydantic import BaseModel, Field


class ServiceLine(BaseModel):
    cpt_code: str
    modifiers: list[str] = []
    charge: float
    units: float = 1.0
    service_date: str | None = None


class ParsedClaim(BaseModel):
    claim_id: str
    sender_id: str = ""
    receiver_id: str = ""
    interchange_control_number: str = ""
    transaction_reference: str = ""
    transaction_date: str = ""

    # Billing provider (NM1*85)
    billing_provider_name: str = ""
    billing_provider_npi: str = ""
    provider_taxonomy: str = ""

    # Rendering provider (NM1*82) — separate from billing
    rendering_provider_name: str = ""
    rendering_provider_npi: str = ""

    # Patient
    patient_first_name: str = ""
    patient_last_name: str = ""
    subscriber_id: str = ""
    patient_dob: str = ""
    patient_gender: str = ""

    # Payer
    payer_name: str = ""
    payer_id: str = ""
    payer_sequence: str = ""
    group_number: str = ""

    # Claim details
    total_charge: float = 0.0
    place_of_service: str = ""
    frequency_code: str = ""
    prior_auth_number: str = ""

    # Diagnosis
    diagnosis_codes: list[str] = []

    # Service lines
    service_lines: list[ServiceLine] = []

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ClaimResponse(BaseModel):
    claim_id: str
    patient_name: str = ""
    payer_name: str = ""
    payer_id: str = ""
    total_charge: float = 0.0
    diagnosis_codes: list[str] = []
    service_lines: list[ServiceLine] = []
    place_of_service: str = ""
    billing_provider_name: str = ""
    billing_provider_npi: str = ""
    patient_dob: str = ""
    patient_gender: str = ""
    provider_taxonomy: str = ""
    prior_auth_number: str = ""
    created_at: datetime | None = None

    # Joined prediction data
    risk_score: float | None = None
    risk_level: str | None = None
    risk_factors: list[dict] | None = None

    # Joined 835 outcome
    actual_outcome: str | None = None
    paid_amount: float | None = None
    carc_codes: list[str] | None = None
    carc_descriptions: list[dict] | None = None


class ClaimListResponse(BaseModel):
    claims: list[ClaimResponse]
    total: int
    skip: int
    limit: int
