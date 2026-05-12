"""
Context-aware claim validation rules.

Produces detailed, human-readable reasons and fix suggestions
per service line and per claim, based on CPT category, payer
conventions, and common billing patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── CPT / HCPCS knowledge base ──────────────────────────────────────────────

# Pattern: 5-digit numeric = CPT, alpha-start = HCPCS Level II
_CPT_RE = re.compile(r"^[0-9]{5}$")
_HCPCS_RE = re.compile(r"^[A-Z][0-9]{4}$")


def classify_issue_type(reason_text: str) -> str | None:
    """Single source of truth for issue type classification."""
    r = reason_text.split("\n")[0].lower()
    if "missing modifier" in r:
        return "missing_modifier"
    if "invalid cpt" in r or "invalid hcpcs" in r:
        return "invalid_cpt"
    if "prior authorization" in r:
        return "missing_prior_auth"
    if "vague" in r and "diagnosis" in r:
        return "vague_diagnosis"
    if "only 1 diagnosis" in r or "weak clinical" in r:
        return "weak_diagnosis"
    if "missing" in r and "npi" in r:
        return "missing_npi"
    if "missing" in r and "date of birth" in r:
        return "missing_dob"
    if "missing" in r and "payer" in r and "identification" in r:
        return "missing_payer_id"
    if "suspicious" in r and "subscriber" in r:
        return "invalid_member_id"
    if "diagnosis/procedure" in r and "mismatch" in r:
        return "dx_cpt_mismatch"
    return None


# Categories by code prefix / range
CPT_CATEGORIES: dict[str, dict] = {
    # E/M
    "em_office": {
        "range": (99201, 99215), "label": "Office E/M visit",
        "typical_modifiers": ["25"],
        "mod_hint": "Add modifier 25 if a significant, separately identifiable E/M service was performed",
    },
    "em_hospital_inpatient": {
        "range": (99221, 99239), "label": "Hospital inpatient E/M",
        "typical_modifiers": ["25"],
        "mod_hint": "Add modifier 25 if reporting E/M with a procedure on the same date",
    },
    "em_er": {
        "range": (99281, 99285), "label": "Emergency department E/M",
        "typical_modifiers": ["25"],
        "mod_hint": "Add modifier 25 for separately identifiable E/M with a procedure",
    },
    "em_consult": {
        "range": (99241, 99255), "label": "Consultation",
        "typical_modifiers": ["25"],
        "mod_hint": "Add modifier 25 if performed with a procedure on the same date",
    },
    # Surgery
    "surgery_musculoskeletal": {
        "range": (29000, 29999), "label": "Musculoskeletal surgery",
        "typical_modifiers": ["LT", "RT", "59"],
        "mod_hint": "Add laterality modifier (LT/RT) and modifier 59 if distinct procedure",
    },
    "surgery_cardio": {
        "range": (33010, 37799), "label": "Cardiovascular surgery",
        "typical_modifiers": ["59", "51"],
        "mod_hint": "Add modifier 59 for distinct procedural service or 51 for multiple procedures",
    },
    # Physical therapy / rehab
    "pt_rehab": {
        "range": (97010, 97799), "label": "Physical therapy / rehab",
        "typical_modifiers": ["59", "GP"],
        "mod_hint": "Add modifier GP (outpatient PT) and 59 if distinct from another service",
    },
    # Radiology
    "radiology_dx": {
        "range": (70010, 76499), "label": "Diagnostic radiology",
        "typical_modifiers": ["26", "TC"],
        "mod_hint": "Add modifier 26 (professional component) or TC (technical component)",
    },
    # Cardiac diagnostics
    "cardiac_dx": {
        "range": (93000, 93799), "label": "Cardiac diagnostic",
        "typical_modifiers": ["26", "TC"],
        "mod_hint": "Add modifier 26 (professional) or TC (technical) as appropriate",
    },
    # Lab / pathology
    "lab": {
        "range": (80000, 89999), "label": "Laboratory / pathology",
        "typical_modifiers": ["91", "59"],
        "mod_hint": "Add modifier 91 for repeat clinical lab test or 59 for distinct test",
    },
    # Venipuncture / blood draw
    "venipuncture": {
        "range": (36400, 36600), "label": "Venipuncture / vascular access",
        "typical_modifiers": ["59"],
        "mod_hint": "Add modifier 59 if distinct from other vascular access procedures",
    },
}

# HCPCS Level II codes — common categories
HCPCS_CATEGORIES: dict[str, dict] = {
    "T1019": {
        "label": "Home health aide / personal care",
        "typical_modifiers": ["HQ", "U1", "GT"],
        "mod_hint": "Most payers require a modifier (e.g., HQ for group, U1 for payer-specific) for this service",
    },
    "G0156": {
        "label": "Home health aide services",
        "typical_modifiers": ["HQ"],
        "mod_hint": "Add modifier HQ or payer-specific modifier for home health aide services",
    },
    "G0151": {
        "label": "PT services in home health",
        "typical_modifiers": ["GP"],
        "mod_hint": "Add modifier GP for physical therapy services under a home health plan of care",
    },
    "G0152": {
        "label": "OT services in home health",
        "typical_modifiers": ["GO"],
        "mod_hint": "Add modifier GO for occupational therapy services under home health",
    },
    "G0153": {
        "label": "SLP services in home health",
        "typical_modifiers": ["GN"],
        "mod_hint": "Add modifier GN for speech-language pathology services",
    },
    "G0299": {
        "label": "Skilled nursing services in home health",
        "typical_modifiers": [],
        "mod_hint": "Verify if payer requires a modifier for skilled nursing home visits",
    },
    "A0425": {
        "label": "Ground mileage ambulance",
        "typical_modifiers": [],
        "mod_hint": "Verify origin/destination modifiers are present",
    },
    "J0585": {
        "label": "Botulinum toxin injection",
        "typical_modifiers": ["59"],
        "mod_hint": "Add modifier 59 if distinct injection site",
    },
}

# Known valid CPT code ranges (broad; anything outside is suspect)
VALID_CPT_RANGES = [
    (10000, 69999),   # Surgery
    (70010, 79999),   # Radiology
    (80000, 89999),   # Path & Lab
    (90281, 99607),   # Medicine + E/M
    (97010, 97799),   # PT/OT/SLP
    (99201, 99499),   # E/M
]


def _is_valid_cpt(code: str) -> bool:
    """Check if a code looks like a valid CPT or HCPCS code."""
    code = code.strip().upper()
    if _HCPCS_RE.match(code):
        return True  # HCPCS Level II — assume valid pattern
    if _CPT_RE.match(code):
        num = int(code)
        return any(lo <= num <= hi for lo, hi in VALID_CPT_RANGES)
    return False


def _get_cpt_info(code: str) -> dict | None:
    """Return category info for a CPT/HCPCS code."""
    code = code.strip().upper()
    # Check HCPCS first (exact match)
    if code in HCPCS_CATEGORIES:
        return HCPCS_CATEGORIES[code]
    # Check CPT ranges
    if _CPT_RE.match(code):
        num = int(code)
        for cat in CPT_CATEGORIES.values():
            lo, hi = cat["range"]
            if lo <= num <= hi:
                return cat
    return None


# ── Member ID validation ─────────────────────────────────────────────────────

_PLACEHOLDER_IDS = {
    "000000000", "111111111", "222222222", "333333333",
    "444444444", "555555555", "666666666", "777777777",
    "888888888", "999999999", "123456789", "000000001",
    "UNKNOWN", "TEST", "TEMP", "NONE", "NA", "N/A", "TBD",
    "SELF", "SELFPAY", "SELF PAY", "CASH",
}

_SUSPICIOUS_PATTERNS = [
    re.compile(r"^(.)\1{5,}$"),        # 6+ repeating chars (AAAAAA, 000000)
    re.compile(r"^TEST", re.I),        # Starts with TEST
    re.compile(r"^TEMP", re.I),        # Starts with TEMP
    re.compile(r"^0{3,}$"),            # All zeros (any length 3+)
    re.compile(r"^X{3,}$", re.I),     # All X's
]


def _is_suspicious_member_id(member_id: str) -> bool:
    """Check if a subscriber/member ID looks like a placeholder or test value."""
    if not member_id or len(member_id) < 2:
        return True  # Missing or too short
    upper = member_id.upper().strip()
    if upper in _PLACEHOLDER_IDS:
        return True
    for pattern in _SUSPICIOUS_PATTERNS:
        if pattern.match(upper):
            return True
    return False


# ── Diagnosis / Procedure mismatch detection ─────────────────────────────────

# ICD-10 chapter prefixes -> compatible CPT categories
_DX_CPT_COMPATIBILITY: dict[str, set[str]] = {
    # Musculoskeletal (M00-M99)
    "M": {"surgery_musculoskeletal", "pt_rehab", "radiology_dx", "em_office",
           "em_hospital_inpatient", "em_er"},
    # Circulatory (I00-I99)
    "I": {"cardiac_dx", "surgery_cardio", "em_office", "em_hospital_inpatient",
           "em_er", "radiology_dx"},
    # Neoplasms (C00-D49)
    "C": {"radiology_dx", "lab", "em_office", "em_hospital_inpatient", "em_er"},
    # Mental/behavioral (F01-F99) — E/M only
    "F": {"em_office", "em_hospital_inpatient", "em_consult"},
    # Pregnancy (O00-O9A)
    "O": {"em_office", "em_hospital_inpatient"},
    # Injury/external causes (S/T)
    "S": {"surgery_musculoskeletal", "em_er", "em_office", "radiology_dx"},
    "T": {"surgery_musculoskeletal", "em_er", "em_office", "radiology_dx"},
    # Screening/factors (Z00-Z99) — broad, only flag if paired with surgery
    "Z": {"em_office", "em_consult", "lab", "radiology_dx"},
}


def _check_dx_cpt_mismatch(diagnosis_codes, service_lines) -> str | None:
    """Check for obvious dx/CPT category mismatches. Returns description or None."""
    if not diagnosis_codes or not service_lines:
        return None

    primary_dx = diagnosis_codes[0].upper().replace(".", "") if diagnosis_codes else ""
    if not primary_dx:
        return None
    dx_chapter = primary_dx[0]

    compatible_categories = _DX_CPT_COMPATIBILITY.get(dx_chapter)
    if compatible_categories is None:
        return None  # Unknown chapter — don't flag

    mismatches = []
    for sl in service_lines:
        code = (sl.cpt_code.strip().upper() if hasattr(sl, "cpt_code")
                else sl.get("cpt_code", "").strip().upper())
        info = _get_cpt_info(code)
        if not info:
            continue
        cpt_cat = None
        for cat_name, cat_info in CPT_CATEGORIES.items():
            if cat_info is info:
                cpt_cat = cat_name
                break
        if cpt_cat and cpt_cat not in compatible_categories:
            mismatches.append(f"{code} ({info['label']}) with dx {primary_dx}")

    if not mismatches:
        return None
    return "Potential mismatch: " + "; ".join(mismatches[:3])


# ── Per-claim issue detection ────────────────────────────────────────────────

@dataclass
class ClaimIssue:
    """A single issue found on a claim, with reason + fix bullets."""
    reason: str          # Human-readable reason paragraph
    fixes: list[str]     # Actionable fix bullet points
    source: str = "static"   # "static" | "adaptive" | "historical_pattern"


@dataclass
class ClaimValidation:
    """Full validation result for one claim."""
    claim_id: str
    patient_name: str
    payer_name: str
    risk_score: float
    risk_level: str
    action: str = ""
    action_label: str = ""
    score_breakdown: dict = field(default_factory=dict)
    issues: list[ClaimIssue] = field(default_factory=list)
    top_factors: list[dict] = field(default_factory=list)
    denial_patterns: list[dict] = field(default_factory=list)


async def validate_claim(
    claim, prediction, session=None, preloaded_patterns=None,
    preloaded_adaptive_rules=None,
) -> ClaimValidation:
    """
    Run all validation rules on a single claim + prediction.

    Phase 1: Collect service line issues into buckets (invalid CPTs, missing
             modifiers grouped by category).
    Phase 2: Emit ONE issue per bucket (not per service line).
    Phase 3: Claim-level checks (prior auth, NPI, DOB, payer, dx, member ID,
             dx/CPT mismatch).
    Phase 4: Enrich fixes with knowledge layer (historical fix reuse) +
             historical denial pattern retrieval.
    Phase 5: Adaptive learned rules from mined denial patterns.

    Args:
        session: Optional DB session for historical pattern queries.
        preloaded_patterns: Pre-fetched DenialPattern list from batch retrieval.
        preloaded_adaptive_rules: Pre-fetched AdaptiveRule list from batch retrieval.
    """
    patient_name = f"{claim.patient_first_name} {claim.patient_last_name}".strip() or "Unknown"
    payer = claim.payer_name or "Unknown"

    issues: list[ClaimIssue] = []

    # ── Phase 1: Collect service line issues into buckets ──
    invalid_cpts: list[str] = []
    # category_label -> {codes: [...], info: {...}}
    missing_mod_by_category: dict[str, dict] = {}
    missing_mod_unknown: list[str] = []

    for sl in claim.service_lines:
        code = sl.cpt_code.strip().upper()
        info = _get_cpt_info(code)
        has_modifier = sl.modifiers and any(m.strip() for m in sl.modifiers)

        # Invalid / unrecognized CPT
        if not _is_valid_cpt(code):
            if code not in invalid_cpts:
                invalid_cpts.append(code)
            continue

        # Missing modifier
        if not has_modifier:
            if info:
                label = info["label"]
                if label not in missing_mod_by_category:
                    missing_mod_by_category[label] = {
                        "codes": [],
                        "info": info,
                    }
                if code not in missing_mod_by_category[label]["codes"]:
                    missing_mod_by_category[label]["codes"].append(code)
            else:
                if code not in missing_mod_unknown:
                    missing_mod_unknown.append(code)

    # ── Phase 2: Emit ONE issue per bucket ──

    # All invalid CPTs -> 1 issue
    if invalid_cpts:
        code_list = ", ".join(invalid_cpts)
        issues.append(ClaimIssue(
            reason=(
                f"Invalid CPT/HCPCS codes: {code_list}\n"
                f"These codes are not recognized as billable"
            ),
            fixes=[
                f"Replace with valid codes from the current CPT/HCPCS code set",
                "Verify each code against the payer's accepted code list",
            ],
        ))

    # Missing modifiers per category -> 1 issue each
    for cat_label, bucket in missing_mod_by_category.items():
        codes = bucket["codes"]
        info = bucket["info"]
        code_list = ", ".join(codes)
        mods = ", ".join(info["typical_modifiers"][:3]) if info["typical_modifiers"] else "payer-specific modifier"
        issues.append(ClaimIssue(
            reason=(
                f"Missing modifier on {cat_label} ({code_list})\n"
                f"{info['mod_hint']}"
            ),
            fixes=[
                f"Add modifier {mods} to: {code_list}",
                f"Verify with {payer}",
            ],
        ))

    # Unknown category modifiers -> 1 combined issue
    if missing_mod_unknown:
        code_list = ", ".join(missing_mod_unknown)
        issues.append(ClaimIssue(
            reason=(
                f"Missing modifier on CPT {code_list}\n"
                f"Most payers require a modifier to process these codes"
            ),
            fixes=[
                f"Add the appropriate modifier for: {code_list}",
                f"Check {payer} billing guidelines for modifier requirements",
            ],
        ))

    # ── Phase 3: Claim-level checks ──

    # Missing prior authorization
    if not claim.prior_auth_number and claim.total_charge > 1000:
        issues.append(ClaimIssue(
            reason=(
                f"No prior authorization on a ${claim.total_charge:,.0f} claim\n"
                f"Most payers require prior auth for claims exceeding $1,000"
            ),
            fixes=[
                f"Obtain prior authorization from {payer} before submitting",
                "Attach the auth number in the CLM segment (REF*G1)",
            ],
        ))

    # Weak / vague diagnosis support
    if len(claim.diagnosis_codes) < 2:
        dx = claim.diagnosis_codes[0] if claim.diagnosis_codes else "none"
        dx_upper = dx.upper().replace(".", "")

        # Check if the sole diagnosis is a known vague/non-specific code
        from app.services.decision_engine import VAGUE_DIAGNOSIS_CODES
        is_vague = dx_upper in VAGUE_DIAGNOSIS_CODES

        if is_vague:
            issues.append(ClaimIssue(
                reason=(
                    f"Vague/non-specific sole diagnosis ({dx})\n"
                    f"This code is non-specific and frequently triggers medical necessity "
                    f"denials, downcoding, and audit flags when used as the only diagnosis"
                ),
                fixes=[
                    f"Replace {dx} with a more specific ICD-10 code that documents the clinical condition",
                    "Add secondary/supporting diagnosis codes to justify medical necessity",
                    "Review clinical documentation for more precise diagnostic coding",
                ],
            ))
        else:
            issues.append(ClaimIssue(
                reason=(
                    f"Only 1 diagnosis code ({dx}) — weak clinical support\n"
                    f"Claims with a single diagnosis have higher denial rates"
                ),
                fixes=[
                    "Add secondary/supporting diagnosis codes to strengthen medical necessity",
                    "Review clinical documentation for additional relevant ICD-10 codes",
                ],
            ))

    # Missing provider NPI
    if not claim.billing_provider_npi:
        issues.append(ClaimIssue(
            reason=(
                "Missing billing provider NPI\n"
                "Claims without a valid NPI are automatically rejected by most clearinghouses"
            ),
            fixes=[
                "Add the 10-digit billing provider NPI in the NM1*85 segment",
                "Verify the NPI is active at nppes.cms.hhs.gov",
            ],
        ))

    # Missing patient DOB
    if not claim.patient_dob:
        issues.append(ClaimIssue(
            reason=(
                "Missing patient date of birth\n"
                "Required for eligibility verification and coordination of benefits"
            ),
            fixes=[
                "Add patient DOB in the DMG segment",
                "Verify DOB matches the payer's enrollment records",
            ],
        ))

    # Missing payer ID
    if not claim.payer_id:
        issues.append(ClaimIssue(
            reason=(
                "Missing payer identification number\n"
                "The payer ID is required for electronic claim routing"
            ),
            fixes=[
                f"Add {payer}'s payer ID in the NM1*PR segment",
                "Look up the payer ID in the clearinghouse enrollment directory",
            ],
        ))

    # Suspicious / placeholder subscriber ID
    subscriber_id = (claim.subscriber_id or "").strip()
    if _is_suspicious_member_id(subscriber_id):
        issues.append(ClaimIssue(
            reason=(
                f"Suspicious subscriber/member ID: '{subscriber_id}'\n"
                f"This ID matches patterns commonly rejected by payers (CARC 31)"
            ),
            fixes=[
                "Verify the subscriber ID against the insurance card",
                "Check eligibility with the payer before submission",
                f"Contact {payer} member services to confirm the correct ID format",
            ],
        ))

    # Diagnosis / procedure category mismatch
    mismatch = _check_dx_cpt_mismatch(claim.diagnosis_codes, claim.service_lines)
    if mismatch:
        issues.append(ClaimIssue(
            reason=(
                f"Diagnosis/procedure category mismatch\n"
                f"{mismatch}"
            ),
            fixes=[
                "Review diagnosis codes for clinical alignment with the billed procedure",
                "Add a more specific diagnosis code supporting the procedure performed",
                "Verify the CPT code matches the documented clinical indication",
            ],
        ))

    # ── Phase 4: Knowledge layer enrichment ──
    primary_cpt = ""
    if claim.service_lines:
        primary_cpt = claim.service_lines[0].cpt_code.strip().upper()

    try:
        from app.services.knowledge_store import get_best_fix
        for iss in issues:
            issue_type = classify_issue_type(iss.reason)
            if not issue_type:
                continue

            best = await get_best_fix(issue_type, payer, primary_cpt)
            if best:
                rate_pct = int(best["success_rate"] * 100)
                iss.fixes.insert(
                    0,
                    f"Recommended fix (worked {rate_pct}% of the time, "
                    f"{best['confidence']} confidence): {best['fix']}",
                )
    except Exception:
        pass  # Knowledge layer is optional; don't break validation

    # Historical denial pattern retrieval
    denial_patterns_data: list[dict] = []
    patterns = preloaded_patterns  # Use pre-fetched if available
    if patterns is None and session is not None:
        try:
            from app.services.denial_pattern_service import get_denial_patterns
            cpt_codes = [sl.cpt_code.strip().upper() for sl in claim.service_lines]
            patterns = await get_denial_patterns(
                session, payer, cpt_codes,
                place_of_service=claim.place_of_service,
            )
        except Exception:
            patterns = None

    if patterns:
        from app.services.denial_pattern_service import pattern_to_dict
        for pattern in patterns:
            denial_patterns_data.append(pattern_to_dict(pattern))
            # Only HIGH severity creates a validation issue
            if pattern.severity == "HIGH" and not pattern.advisory_only:
                carc_list = ", ".join(
                    f"{c['code']} ({c['description']})"
                    for c in pattern.top_carc_codes[:3]
                )
                fix_list = [f["fix"] for f in pattern.recommended_fixes[:2]]
                issues.append(ClaimIssue(
                    reason=(
                        f"Historical denial pattern: {payer} + CPT {pattern.cpt_code} — "
                        f"{int(pattern.denial_rate * 100)}% denial rate "
                        f"({pattern.sample_count} claims)\n"
                        f"Common denial codes: {carc_list}"
                    ),
                    fixes=fix_list or [
                        "Review claim against payer-specific guidelines before submission"
                    ],
                ))

    # ── Phase 5: Adaptive learned rules ──
    adaptive_rules = preloaded_adaptive_rules
    if adaptive_rules is None and session is not None:
        try:
            from app.repositories import adaptive_rule_repo
            cpt_codes_list = [sl.cpt_code.strip().upper() for sl in claim.service_lines]
            adaptive_rules = await adaptive_rule_repo.get_active_rules_for_payer(
                session, payer, cpt_codes_list,
            )
        except Exception:
            adaptive_rules = None

    if adaptive_rules:
        claim_cpts = {sl.cpt_code.strip().upper() for sl in claim.service_lines}
        claim_dxs = set(claim.diagnosis_codes or [])
        for rule in adaptive_rules:
            if rule.operator_approved is False:
                continue
            if rule.cpt_code and rule.cpt_code not in claim_cpts:
                continue
            if rule.diagnosis_code and rule.diagnosis_code not in claim_dxs:
                continue

            if rule.severity == "ERROR" and rule.confidence_level == "HIGH":
                issues.append(ClaimIssue(
                    reason=f"[Learned] {rule.rule_description}\n"
                           f"Based on {rule.denied_claims}/{rule.total_claims} denials "
                           f"({int(rule.denial_rate * 100)}% rate, {rule.confidence_level} confidence)",
                    fixes=[rule.fix_suggestion],
                    source="adaptive",
                ))
            elif rule.severity == "WARNING" and rule.confidence_level in ("MEDIUM", "HIGH"):
                issues.append(ClaimIssue(
                    reason=f"[Advisory] {rule.rule_description}\n"
                           f"Based on {rule.denied_claims}/{rule.total_claims} denials "
                           f"({int(rule.denial_rate * 100)}% rate, {rule.confidence_level} confidence)",
                    fixes=[rule.fix_suggestion],
                    source="adaptive",
                ))

    # Top factors from prediction
    top_factors = [
        {"name": f.display_name, "impact": f.impact}
        for f in (prediction.risk_factors or [])[:3]
    ]

    return ClaimValidation(
        claim_id=prediction.claim_id,
        patient_name=patient_name,
        payer_name=payer,
        risk_score=prediction.risk_score,
        risk_level=prediction.risk_level,
        issues=issues,
        top_factors=top_factors,
        denial_patterns=denial_patterns_data,
    )


def _suggest_replacement(code: str, info: dict | None) -> str:
    """Suggest a replacement for an invalid code."""
    if info:
        return info.get("label", "a valid CPT code")
    # Heuristic based on code pattern
    c = code.upper()
    if c.startswith("T"):
        return "T1019, G0156"
    if c.startswith("G"):
        return "G0299, G0156"
    if c.startswith("J"):
        return "a valid J-code drug code"
    return "a valid CPT from the current code set"


def validation_to_dict(v: ClaimValidation) -> dict:
    """Convert a ClaimValidation to the API response dict."""
    result = {
        "claim_id": v.claim_id,
        "patient_name": v.patient_name,
        "payer_name": v.payer_name,
        "risk_score": v.risk_score,
        "risk_level": v.risk_level,
        "action": v.action,
        "action_label": v.action_label,
        "score_breakdown": v.score_breakdown,
        "issues": [
            {"reason": iss.reason, "fixes": iss.fixes, "source": iss.source}
            for iss in v.issues
        ],
        "top_factors": v.top_factors,
        "denial_patterns": v.denial_patterns,
    }
    return result
