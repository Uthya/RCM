"""
Pre-storage data validation for parsed claims.

Runs between EDI parser output and MongoDB storage to catch
data quality issues before they pollute features, predictions,
and training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from app.schemas.claim import ParsedClaim

logger = structlog.get_logger()

_NPI_RE = re.compile(r"^\d{10}$")
_DATE_YYYYMMDD_RE = re.compile(r"^\d{8}$")


@dataclass
class ValidationResult:
    """Result of pre-storage validation for a batch of parsed claims."""
    valid_claims: list[ParsedClaim] = field(default_factory=list)
    rejected_claims: list[dict] = field(default_factory=list)   # {claim_id, reasons: [str]}
    warnings: list[dict] = field(default_factory=list)          # {claim_id, warnings: [str]}
    duplicate_ids: list[str] = field(default_factory=list)


def validate_parsed_claims(claims: list[ParsedClaim]) -> ValidationResult:
    """Pre-storage validation. Catches data quality issues before DB.

    Rules:
    - Claims failing required field checks -> rejected (not stored)
    - Claims with format issues -> warned but stored
    - Duplicate claim_ids -> keep first, reject subsequent
    """
    result = ValidationResult()
    seen_ids: dict[str, int] = {}  # claim_id -> index of first occurrence

    for i, claim in enumerate(claims):
        reasons: list[str] = []
        claim_warnings: list[str] = []

        # ── Duplicate detection ──
        if claim.claim_id in seen_ids:
            result.duplicate_ids.append(claim.claim_id)
            result.rejected_claims.append({
                "claim_id": claim.claim_id,
                "reasons": [f"Duplicate claim_id (first seen at position {seen_ids[claim.claim_id]})"],
            })
            continue

        seen_ids[claim.claim_id] = i

        # ── Required field checks (rejection) ──
        if not claim.claim_id or not claim.claim_id.strip():
            reasons.append("Missing or empty claim_id")

        if not claim.service_lines:
            reasons.append("No service lines")

        # ── Format validation (warnings, non-blocking) ──
        if claim.billing_provider_npi and not _NPI_RE.match(claim.billing_provider_npi):
            claim_warnings.append(
                f"Billing provider NPI '{claim.billing_provider_npi}' is not 10 digits"
            )

        if claim.rendering_provider_npi and not _NPI_RE.match(claim.rendering_provider_npi):
            claim_warnings.append(
                f"Rendering provider NPI '{claim.rendering_provider_npi}' is not 10 digits"
            )

        if claim.patient_dob and not _DATE_YYYYMMDD_RE.match(claim.patient_dob):
            claim_warnings.append(
                f"Patient DOB '{claim.patient_dob}' is not in YYYYMMDD format"
            )

        if claim.transaction_date and not _DATE_YYYYMMDD_RE.match(claim.transaction_date):
            claim_warnings.append(
                f"Transaction date '{claim.transaction_date}' is not in YYYYMMDD format"
            )

        # Check service line dates
        for j, sl in enumerate(claim.service_lines):
            if sl.service_date and not _DATE_YYYYMMDD_RE.match(sl.service_date):
                claim_warnings.append(
                    f"Service line {j + 1} date '{sl.service_date}' is not in YYYYMMDD format"
                )

        # ── Soft warnings (non-blocking) ──
        if not claim.patient_first_name and not claim.patient_last_name:
            claim_warnings.append("Missing patient name")

        if not claim.payer_id:
            claim_warnings.append("Missing payer_id")

        if claim.total_charge == 0:
            claim_warnings.append("Total charge is zero")

        # ── Decide: reject or accept ──
        if reasons:
            result.rejected_claims.append({
                "claim_id": claim.claim_id or f"<empty at position {i}>",
                "reasons": reasons,
            })
        else:
            result.valid_claims.append(claim)
            if claim_warnings:
                result.warnings.append({
                    "claim_id": claim.claim_id,
                    "warnings": claim_warnings,
                })

    if result.rejected_claims or result.duplicate_ids:
        logger.warning(
            "Pre-storage validation completed with issues",
            total=len(claims),
            valid=len(result.valid_claims),
            rejected=len(result.rejected_claims),
            duplicates=len(result.duplicate_ids),
        )

    return result
