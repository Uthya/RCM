"""SQLAlchemy ORM models for RCM AI — 16 tables with PostgreSQL-native features."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, ENUM as PgEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# PostgreSQL ENUM types
# ---------------------------------------------------------------------------

class AttemptType(str, enum.Enum):
    ORIGINAL = "ORIGINAL"
    RESUBMISSION = "RESUBMISSION"
    CORRECTED = "CORRECTED"
    APPEAL = "APPEAL"


class ConfidenceLevel(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ClaimStatus(str, enum.Enum):
    paid = "paid"
    denied = "denied"
    partial = "partial"


class OutcomeStatus(str, enum.Enum):
    paid = "paid"
    denied = "denied"
    partial = "partial"
    pending = "pending"


class ModelStatus(str, enum.Enum):
    trained = "trained"
    candidate = "candidate"
    shadow = "shadow"
    active = "active"
    retired = "retired"
    rolled_back = "rolled_back"


class LifecycleStatus(str, enum.Enum):
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    DENIED = "DENIED"
    RESUBMITTED = "RESUBMITTED"
    PAID = "PAID"
    PARTIAL = "PARTIAL"
    APPEAL = "APPEAL"
    VOID = "VOID"


class AttemptStatus(str, enum.Enum):
    PENDING = "PENDING"
    DENIED = "DENIED"
    PAID = "PAID"
    PARTIAL = "PARTIAL"
    VOID = "VOID"


class FixOutcome(str, enum.Enum):
    paid = "paid"
    denied = "denied"
    partial = "partial"


# ---------------------------------------------------------------------------
# 1. claims
# ---------------------------------------------------------------------------

class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    sender_id: Mapped[str | None] = mapped_column(String(64))
    receiver_id: Mapped[str | None] = mapped_column(String(64))
    interchange_control_number: Mapped[str | None] = mapped_column(String(64))
    transaction_reference: Mapped[str | None] = mapped_column(String(64))
    transaction_date: Mapped[str | None] = mapped_column(String(32))
    billing_provider_name: Mapped[str | None] = mapped_column(String(256))
    billing_provider_npi: Mapped[str | None] = mapped_column(String(20))
    rendering_provider_name: Mapped[str | None] = mapped_column(String(256))
    rendering_provider_npi: Mapped[str | None] = mapped_column(String(20))
    patient_first_name: Mapped[str | None] = mapped_column(String(128))
    patient_last_name: Mapped[str | None] = mapped_column(String(128))
    patient_dob: Mapped[str | None] = mapped_column(String(16))
    patient_gender: Mapped[str | None] = mapped_column(String(4))
    subscriber_id: Mapped[str | None] = mapped_column(String(64))
    payer_name: Mapped[str | None] = mapped_column(String(256))
    payer_id: Mapped[str | None] = mapped_column(String(64), index=True)
    payer_sequence: Mapped[str | None] = mapped_column(String(4))
    group_number: Mapped[str | None] = mapped_column(String(64))
    total_charge: Mapped[float] = mapped_column(Float, default=0.0)
    place_of_service: Mapped[str | None] = mapped_column(String(8))
    frequency_code: Mapped[str | None] = mapped_column(String(4))
    prior_auth_number: Mapped[str | None] = mapped_column(String(64))
    provider_taxonomy: Mapped[str | None] = mapped_column(String(32))
    diagnosis_codes = mapped_column(ARRAY(String), default=list)
    service_lines = mapped_column(JSONB, default=list)
    validation_issues = mapped_column(JSONB, default=list)
    issue_count: Mapped[int] = mapped_column(Integer, default=0)
    action: Mapped[str | None] = mapped_column(String(32))
    action_label: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# ---------------------------------------------------------------------------
# 2. claim_outcomes
# ---------------------------------------------------------------------------

class ClaimOutcome(Base):
    __tablename__ = "claim_outcomes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    attempt_type: Mapped[str | None] = mapped_column(String(16))
    outcome_status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    paid_amount: Mapped[float] = mapped_column(Float, default=0.0)
    carc_codes = mapped_column(ARRAY(String), default=list)
    carc_descriptions = mapped_column(JSONB, default=list)
    model_version: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        UniqueConstraint("claim_id", "attempt_number", name="uq_claim_outcomes_claim_attempt"),
    )


# ---------------------------------------------------------------------------
# 3. remittances
# ---------------------------------------------------------------------------

class Remittance(Base):
    __tablename__ = "remittances"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payer_control_number: Mapped[str | None] = mapped_column(String(64))
    claim_status_code: Mapped[str | None] = mapped_column(String(8))
    claim_status: Mapped[str | None] = mapped_column(String(16))
    billed_amount: Mapped[float] = mapped_column(Float, default=0.0)
    paid_amount: Mapped[float] = mapped_column(Float, default=0.0)
    patient_responsibility: Mapped[float] = mapped_column(Float, default=0.0)
    payer_name: Mapped[str | None] = mapped_column(String(256))
    payee_name: Mapped[str | None] = mapped_column(String(256))
    payee_npi: Mapped[str | None] = mapped_column(String(20))
    total_payment_amount: Mapped[float | None] = mapped_column(Float)
    payment_method: Mapped[str | None] = mapped_column(String(16))
    payment_date: Mapped[str | None] = mapped_column(String(32))
    trace_number: Mapped[str | None] = mapped_column(String(64))
    adjustments = mapped_column(JSONB, default=list)
    carc_codes = mapped_column(ARRAY(String), default=list)
    rarc_codes = mapped_column(ARRAY(String), default=list)
    service_lines = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index("ix_remittances_payer_status", "payer_name", "claim_status"),
    )


# ---------------------------------------------------------------------------
# 4. predictions
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    attempt_type: Mapped[str | None] = mapped_column(String(16))
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    features = mapped_column(JSONB, default=dict)
    risk_factors = mapped_column(JSONB, default=list)
    feature_version: Mapped[str | None] = mapped_column(String(16))
    feature_count: Mapped[int | None] = mapped_column(Integer)
    feature_hash: Mapped[str | None] = mapped_column(String(32))
    model_version: Mapped[str | None] = mapped_column(String(16))
    action: Mapped[str | None] = mapped_column(String(32))
    action_label: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("claim_id", "attempt_number", name="uq_predictions_claim_attempt"),
    )


# ---------------------------------------------------------------------------
# 5. ml_training_data
# ---------------------------------------------------------------------------

class MLTrainingData(Base):
    __tablename__ = "ml_training_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    is_first_attempt: Mapped[bool] = mapped_column(Boolean, default=True)
    features = mapped_column(JSONB, default=dict)
    label: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    actual_outcome: Mapped[str | None] = mapped_column(String(16))
    denial_code: Mapped[str | None] = mapped_column(String(16))
    denial_code_description: Mapped[str | None] = mapped_column(Text)
    all_carc_codes = mapped_column(ARRAY(String), default=list)
    paid_amount: Mapped[float | None] = mapped_column(Float)
    billed_amount: Mapped[float | None] = mapped_column(Float)
    model_version_at_prediction: Mapped[str | None] = mapped_column(String(16))
    prediction_risk_score: Mapped[float | None] = mapped_column(Float)
    feature_version: Mapped[str | None] = mapped_column(String(16), index=True)
    feature_count: Mapped[int | None] = mapped_column(Integer)
    feature_hash: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        UniqueConstraint("claim_id", "attempt_number", name="uq_ml_training_data_claim_attempt"),
    )


# ---------------------------------------------------------------------------
# 6. ml_training_data_archive
# ---------------------------------------------------------------------------

class MLTrainingDataArchive(Base):
    __tablename__ = "ml_training_data_archive"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    is_first_attempt: Mapped[bool] = mapped_column(Boolean, default=True)
    features = mapped_column(JSONB, default=dict)
    label: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    actual_outcome: Mapped[str | None] = mapped_column(String(16))
    denial_code: Mapped[str | None] = mapped_column(String(16))
    denial_code_description: Mapped[str | None] = mapped_column(Text)
    all_carc_codes = mapped_column(ARRAY(String), default=list)
    paid_amount: Mapped[float | None] = mapped_column(Float)
    billed_amount: Mapped[float | None] = mapped_column(Float)
    model_version_at_prediction: Mapped[str | None] = mapped_column(String(16))
    prediction_risk_score: Mapped[float | None] = mapped_column(Float)
    feature_version: Mapped[str | None] = mapped_column(String(16))
    feature_count: Mapped[int | None] = mapped_column(Integer)
    feature_hash: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# ---------------------------------------------------------------------------
# 7. model_registry
# ---------------------------------------------------------------------------

class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    version_str: Mapped[str | None] = mapped_column(String(16))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str | None] = mapped_column(String(16))
    model_path: Mapped[str | None] = mapped_column(String(512))
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    real_samples: Mapped[int | None] = mapped_column(Integer)
    synthetic_samples: Mapped[int | None] = mapped_column(Integer)
    feature_version: Mapped[str | None] = mapped_column(String(16))
    feature_count: Mapped[int | None] = mapped_column(Integer)
    feature_hash: Mapped[str | None] = mapped_column(String(32))
    metrics = mapped_column(JSONB, default=dict)
    feature_importance = mapped_column(JSONB, default=dict)
    top_denial_codes = mapped_column(JSONB, default=list)


# ---------------------------------------------------------------------------
# 8. claim_lifecycle
# ---------------------------------------------------------------------------

class ClaimLifecycle(Base):
    __tablename__ = "claim_lifecycle"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    payer_name: Mapped[str | None] = mapped_column(String(256), index=True)
    patient_name: Mapped[str | None] = mapped_column(String(256))
    current_status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    total_attempts: Mapped[int] = mapped_column(Integer, default=1, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    attempts: Mapped[list["LifecycleAttempt"]] = relationship(
        back_populates="lifecycle",
        cascade="all, delete-orphan",
        order_by="LifecycleAttempt.attempt_number",
    )


# ---------------------------------------------------------------------------
# 9. lifecycle_attempts
# ---------------------------------------------------------------------------

class LifecycleAttempt(Base):
    __tablename__ = "lifecycle_attempts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    lifecycle_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("claim_lifecycle.id", ondelete="CASCADE"), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_type: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="PENDING")
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    frequency_code: Mapped[str | None] = mapped_column(String(4))
    fix_applied: Mapped[str | None] = mapped_column(Text)
    features = mapped_column(JSONB, default=dict)
    service_lines = mapped_column(JSONB, default=list)
    validation_issues = mapped_column(JSONB, default=list)
    fixes_recommended = mapped_column(JSONB, default=list)
    prediction_risk_score: Mapped[float | None] = mapped_column(Float)
    prediction_risk_level: Mapped[str | None] = mapped_column(String(8))
    model_version: Mapped[str | None] = mapped_column(String(16))
    denial_codes = mapped_column(ARRAY(String), default=list)
    paid_amount: Mapped[float] = mapped_column(Float, default=0.0)
    billed_amount: Mapped[float] = mapped_column(Float, default=0.0)
    remittance_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    lifecycle: Mapped["ClaimLifecycle"] = relationship(back_populates="attempts")

    __table_args__ = (
        UniqueConstraint("lifecycle_id", "attempt_number", name="uq_lifecycle_attempts_lc_attempt"),
    )


# ---------------------------------------------------------------------------
# 10. fix_history
# ---------------------------------------------------------------------------

class FixHistory(Base):
    __tablename__ = "fix_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt_number: Mapped[int | None] = mapped_column(Integer)
    issue_type: Mapped[str] = mapped_column(String(64), nullable=False)
    fix_applied: Mapped[str] = mapped_column(Text, nullable=False)
    payer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    cpt_code: Mapped[str | None] = mapped_column(String(16))
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index("ix_fix_history_payer_issue", "payer_name", "issue_type"),
        Index("ix_fix_history_composite", "payer_name", "cpt_code", "issue_type", "fix_applied"),
    )


# ---------------------------------------------------------------------------
# 11. fix_effectiveness
# ---------------------------------------------------------------------------

class FixEffectiveness(Base):
    __tablename__ = "fix_effectiveness"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    payer_name: Mapped[str] = mapped_column(String(256), nullable=False)
    cpt_code: Mapped[str] = mapped_column(String(16), nullable=False)
    issue_type: Mapped[str] = mapped_column(String(64), nullable=False)
    fix_applied: Mapped[str] = mapped_column(Text, nullable=False)
    total: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[int] = mapped_column(Integer, default=0)
    failure: Mapped[int] = mapped_column(Integer, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    confidence_level: Mapped[str] = mapped_column(String(8), default="LOW")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("payer_name", "cpt_code", "issue_type", "fix_applied", name="uq_fix_effectiveness_composite"),
    )


# ---------------------------------------------------------------------------
# 12. shadow_predictions
# ---------------------------------------------------------------------------

class ShadowPrediction(Base):
    __tablename__ = "shadow_predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(String(64), nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    active_score: Mapped[float] = mapped_column(Float, nullable=False)
    shadow_score: Mapped[float] = mapped_column(Float, nullable=False)
    active_version: Mapped[str | None] = mapped_column(String(16))
    shadow_version: Mapped[str | None] = mapped_column(String(16))
    actual_outcome: Mapped[str | None] = mapped_column(String(16))
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("claim_id", "attempt_number", name="uq_shadow_predictions_claim_attempt"),
    )


# ---------------------------------------------------------------------------
# 13. upload_history
# ---------------------------------------------------------------------------

class UploadHistory(Base):
    __tablename__ = "upload_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    filename: Mapped[str | None] = mapped_column(String(512))
    file_type: Mapped[str | None] = mapped_column(String(8))
    claim_count: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# ---------------------------------------------------------------------------
# 14. training_history
# ---------------------------------------------------------------------------

class TrainingHistory(Base):
    __tablename__ = "training_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_version: Mapped[str | None] = mapped_column(String(16))
    real_samples: Mapped[int | None] = mapped_column(Integer)
    synthetic_samples: Mapped[int | None] = mapped_column(Integer)
    total_samples: Mapped[int | None] = mapped_column(Integer)
    denied_count: Mapped[int | None] = mapped_column(Integer)
    paid_count: Mapped[int | None] = mapped_column(Integer)
    denial_rate: Mapped[float | None] = mapped_column(Float)
    training_window_days: Mapped[int | None] = mapped_column(Integer)
    used_full_dataset: Mapped[bool | None] = mapped_column(Boolean)
    metrics = mapped_column(JSONB, default=dict)
    feature_importance = mapped_column(JSONB, default=dict)
    top_denial_codes = mapped_column(JSONB, default=list)
    elapsed_seconds: Mapped[float | None] = mapped_column(Float)


# ---------------------------------------------------------------------------
# 15. decision_config
# ---------------------------------------------------------------------------

class DecisionConfig(Base):
    __tablename__ = "decision_config"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    weights = mapped_column(JSONB, default=dict)


# ---------------------------------------------------------------------------
# 16. cpt_risk_config
# ---------------------------------------------------------------------------

class CptRiskConfig(Base):
    __tablename__ = "cpt_risk_config"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    cpt_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    label: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("cpt_prefix", name="uq_cpt_risk_config_prefix"),
    )


# ---------------------------------------------------------------------------
# 17. adaptive_rules
# ---------------------------------------------------------------------------

class AdaptiveRule(Base):
    __tablename__ = "adaptive_rules"

    id = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Rule identity (composite unique key)
    rule_type        = mapped_column(String(64), nullable=False)
    payer_name       = mapped_column(String(256), nullable=False)
    cpt_code         = mapped_column(String(16), nullable=False, default="")
    carc_code        = mapped_column(String(16), nullable=False, default="")
    diagnosis_code   = mapped_column(String(16), nullable=False, default="")

    # Rule content
    rule_description = mapped_column(Text, nullable=False)
    fix_suggestion   = mapped_column(Text, nullable=False)
    issue_type       = mapped_column(String(64), nullable=False)

    # Evidence
    total_claims     = mapped_column(Integer, default=0)
    denied_claims    = mapped_column(Integer, default=0)
    denial_rate      = mapped_column(Float, default=0.0)

    # Confidence lifecycle
    confidence_level = mapped_column(String(8), default="LOW")
    severity         = mapped_column(String(8), default="INFO")
    is_active        = mapped_column(Boolean, default=True)
    threshold_value  = mapped_column(Float, nullable=True)

    # Timestamps
    last_mined_at    = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at       = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at       = mapped_column(DateTime(timezone=True), server_default=func.now())
    retired_at       = mapped_column(DateTime(timezone=True), nullable=True)

    # Operator overrides
    operator_approved = mapped_column(Boolean, nullable=True)
    operator_notes    = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("rule_type", "payer_name", "cpt_code", "carc_code", "diagnosis_code",
                         name="uq_adaptive_rules_identity"),
        Index("ix_adaptive_rules_payer_active", "payer_name", "is_active"),
        Index("ix_adaptive_rules_confidence", "confidence_level", "severity"),
    )
