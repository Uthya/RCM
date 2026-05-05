"""
Decision Engine — single source of truth for claim action routing.

Computes a composite score from ML model output + payer weight + CPT risk
pattern + issue severity, then routes to auto_submit / review / fix_required.

Weights are loaded from MongoDB at startup (falls back to defaults).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

# ── Default config (used if DB collections are empty) ──

DEFAULT_PAYER_WEIGHTS: dict[str, float] = {
    "MEDICARE": 0.05,
    "MEDICAID": 0.05,
    "TRICARE": 0.04,
    "AETNA": 0.03,
    "UNITEDHEALTH": 0.03,
}

DEFAULT_CPT_RISK_PATTERNS: list[dict] = [
    {"cpt_prefix": "9921", "weight": 0.02, "label": "E/M High", "reason": "high-level E/M audit risk"},
    {"cpt_prefix": "992", "weight": 0.03, "label": "E/M", "reason": "frequent modifier issues"},
    {"cpt_prefix": "270", "weight": 0.02, "label": "Radiology", "reason": "auth-heavy"},
    {"cpt_prefix": "977", "weight": 0.02, "label": "PT/Rehab", "reason": "documentation-heavy"},
]

# ── Critical issue weights ──
# These issues are individually significant enough to affect the score
# regardless of total issue count. They reflect real-world rejection reasons.
CRITICAL_ISSUE_WEIGHTS: dict[str, float] = {
    "missing_npi": 0.25,             # auto-rejected by every clearinghouse
    "missing_modifier": 0.15,        # most common denial reason
    "missing_prior_auth": 0.10,      # high-charge claims without auth
    "invalid_cpt": 0.20,             # unbillable — guaranteed rejection
    "missing_payer_id": 0.15,        # cannot route the claim
    "weak_diagnosis": 0.10,          # medical necessity / downcoding risk
    "vague_sole_diagnosis": 0.15,    # non-specific ICD as sole dx — audit flag
    "missing_dob": 0.05,             # eligibility verification fails
    "concentration_risk": 0.08,      # same patient high-volume billing pattern
}

# ── Payer-specific modifier escalation ──
# Some payers (especially government) deny 100% of claims with missing modifiers.
# These multipliers are applied to the missing_modifier weight for those payers.
PAYER_MODIFIER_MULTIPLIER: dict[str, float] = {
    "MEDICAID": 2.0,     # Medicaid denies/pends ~100% without modifiers
    "MA": 2.0,           # Medicaid alias
    "MEDICARE": 1.5,     # Medicare strict on modifier compliance
    "TRICARE": 1.5,      # Government — strict modifier enforcement
}

# ── Non-specific / vague ICD-10 codes ──
# These codes as the sole diagnosis raise medical necessity and audit flags.
VAGUE_DIAGNOSIS_CODES: set[str] = {
    "R69",    # Illness, unspecified
    "Z741",   # Need for assistance with personal care (social code)
    "Z739",   # Problem related to life management, unspecified
    "Z762",   # Encounter for health supervision of foundling
    "Z711",   # Person with feared health complaint
    "R688",   # Other general symptoms and signs
    "R6889",  # Other general symptoms and signs
    "Z7689",  # Persons encountering health services in other circumstances
}


@dataclass
class ScoreBreakdown:
    """Transparent breakdown of composite score."""
    base_score: float
    payer_weight: float
    payer_name: str
    cpt_weight: float
    cpt_label: str
    issue_weight: float
    final_score: float
    issue_details: list[dict] = field(default_factory=list)


@dataclass
class ClaimDecision:
    action: str            # "auto_submit" | "review" | "fix_required"
    action_label: str      # "Auto Submit" | "Review" | "Fix Required"
    can_submit: bool
    requires_review: bool
    threshold_note: str
    score_breakdown: ScoreBreakdown


# ── In-memory config (loaded from DB at startup) ──

_payer_weights: dict[str, float] = {}
_cpt_patterns: list[dict] = []
_config_loaded: bool = False


async def load_config(db) -> None:
    """Load decision config from DB. Falls back to defaults if empty."""
    global _payer_weights, _cpt_patterns, _config_loaded

    try:
        pw = await db.decision_config.find_one({"type": "payer_weights"})
        _payer_weights = pw["weights"] if pw else dict(DEFAULT_PAYER_WEIGHTS)
    except Exception:
        _payer_weights = dict(DEFAULT_PAYER_WEIGHTS)

    try:
        cpt_docs = await db.cpt_risk_config.find().to_list(100)
        _cpt_patterns = cpt_docs if cpt_docs else list(DEFAULT_CPT_RISK_PATTERNS)
    except Exception:
        _cpt_patterns = list(DEFAULT_CPT_RISK_PATTERNS)

    _config_loaded = True
    logger.info(
        "Decision engine config loaded",
        payer_count=len(_payer_weights),
        cpt_pattern_count=len(_cpt_patterns),
    )


def _ensure_defaults() -> None:
    """Ensure in-memory config has at least defaults if load_config wasn't called."""
    global _payer_weights, _cpt_patterns, _config_loaded
    if not _config_loaded:
        _payer_weights = dict(DEFAULT_PAYER_WEIGHTS)
        _cpt_patterns = list(DEFAULT_CPT_RISK_PATTERNS)
        _config_loaded = True


def classify_issues(issues: list[dict], payer_name: str = "") -> list[dict]:
    """Classify validation issues into weighted categories.

    Each issue dict should have a 'reason' key with the first line
    identifying the issue type. Payer name is used for payer-specific
    weight escalation (e.g., Medicaid modifier strictness).
    """
    payer_key = payer_name.upper().split()[0] if payer_name else ""
    modifier_multiplier = PAYER_MODIFIER_MULTIPLIER.get(payer_key, 1.0)

    classified = []
    for iss in issues:
        reason = iss.get("reason", "")
        first_line = reason.split("\n")[0].lower()

        if "missing billing provider npi" in first_line or "missing provider npi" in first_line:
            classified.append({"type": "missing_npi", "weight": CRITICAL_ISSUE_WEIGHTS["missing_npi"]})
        elif "missing modifier" in first_line:
            weight = round(CRITICAL_ISSUE_WEIGHTS["missing_modifier"] * modifier_multiplier, 4)
            classified.append({"type": "missing_modifier", "weight": weight,
                               "payer_escalated": modifier_multiplier > 1.0})
        elif "invalid cpt" in first_line or "invalid hcpcs" in first_line:
            classified.append({"type": "invalid_cpt", "weight": CRITICAL_ISSUE_WEIGHTS["invalid_cpt"]})
        elif "prior authorization" in first_line or "no prior auth" in first_line:
            classified.append({"type": "missing_prior_auth", "weight": CRITICAL_ISSUE_WEIGHTS["missing_prior_auth"]})
        elif "missing payer" in first_line:
            classified.append({"type": "missing_payer_id", "weight": CRITICAL_ISSUE_WEIGHTS["missing_payer_id"]})
        elif "vague" in first_line or "non-specific" in first_line:
            classified.append({"type": "vague_sole_diagnosis", "weight": CRITICAL_ISSUE_WEIGHTS["vague_sole_diagnosis"]})
        elif "diagnosis" in first_line and ("weak" in first_line or "only 1" in first_line):
            classified.append({"type": "weak_diagnosis", "weight": CRITICAL_ISSUE_WEIGHTS["weak_diagnosis"]})
        elif "concentration risk" in first_line or "high-volume" in first_line:
            classified.append({"type": "concentration_risk", "weight": CRITICAL_ISSUE_WEIGHTS["concentration_risk"]})
        elif "missing patient date" in first_line or "missing patient dob" in first_line:
            classified.append({"type": "missing_dob", "weight": CRITICAL_ISSUE_WEIGHTS["missing_dob"]})
        else:
            classified.append({"type": "other", "weight": 0.03})

    return classified


def decide(
    risk_score: float,
    issue_count: int = 0,
    payer_name: str = "",
    primary_cpt: str = "",
    issues: list[dict] | None = None,
) -> ClaimDecision:
    """
    Compute composite score and return action decision.

    Composite = base ML score + payer weight + CPT risk weight + issue severity.
    Capped at 1.0.

    If `issues` (list of issue dicts with 'reason' key) is provided, uses
    per-issue severity weights. Otherwise falls back to count-based escalation.
    """
    _ensure_defaults()

    # 1. Payer weight
    payer_key = payer_name.upper().split()[0] if payer_name else ""
    payer_w = _payer_weights.get(payer_key, 0.0)

    # 2. CPT risk weight — longest prefix match first
    cpt_w = 0.0
    cpt_label = ""
    for pat in sorted(_cpt_patterns, key=lambda p: len(p.get("cpt_prefix", "")), reverse=True):
        prefix = pat.get("cpt_prefix", "")
        if prefix and primary_cpt.startswith(prefix):
            cpt_w = pat.get("weight", 0.0)
            cpt_label = pat.get("label", "")
            break

    # 3. Issue severity — per-issue weights if available, else count-based
    issue_details: list[dict] = []
    if issues is not None and len(issues) > 0:
        issue_details = classify_issues(issues, payer_name=payer_name)
        # Sum individual weights — each critical issue contributes its own weight
        issue_w = sum(d["weight"] for d in issue_details)
    else:
        # Fallback to count-based escalation
        if issue_count >= 5:
            issue_w = 0.15
        elif issue_count >= 3:
            issue_w = 0.08
        elif issue_count >= 1:
            issue_w = 0.04
        else:
            issue_w = 0.0

    # 4. Composite (capped at 1.0)
    final = min(risk_score + payer_w + cpt_w + issue_w, 1.0)

    breakdown = ScoreBreakdown(
        base_score=round(risk_score, 4),
        payer_weight=payer_w,
        payer_name=payer_key,
        cpt_weight=cpt_w,
        cpt_label=cpt_label,
        issue_weight=round(issue_w, 4),
        final_score=round(final, 4),
        issue_details=issue_details,
    )

    # 5. Route action
    has_critical = any(
        d["type"] in ("missing_npi", "invalid_cpt", "missing_payer_id")
        or (d["type"] == "missing_modifier" and d.get("payer_escalated"))
        for d in issue_details
    )

    if final < 0.3 and issue_count == 0:
        return ClaimDecision(
            action="auto_submit",
            action_label="Auto Submit",
            can_submit=True,
            requires_review=False,
            threshold_note="Score below 0.3 with no issues",
            score_breakdown=breakdown,
        )
    elif final > 0.7 or issue_count > 3 or has_critical:
        return ClaimDecision(
            action="fix_required",
            action_label="Fix Required",
            can_submit=False,
            requires_review=False,
            threshold_note="Score above 0.7, critical issue, or more than 3 issues",
            score_breakdown=breakdown,
        )
    else:
        return ClaimDecision(
            action="review",
            action_label="Review",
            can_submit=True,
            requires_review=True,
            threshold_note="Score between 0.3-0.7, manual review recommended",
            score_breakdown=breakdown,
        )


def decision_to_dict(d: ClaimDecision) -> dict:
    """Serialize a ClaimDecision to a plain dict for API responses / MongoDB."""
    return {
        "action": d.action,
        "action_label": d.action_label,
        "can_submit": d.can_submit,
        "requires_review": d.requires_review,
        "threshold_note": d.threshold_note,
        "score_breakdown": {
            "base_score": d.score_breakdown.base_score,
            "payer_weight": d.score_breakdown.payer_weight,
            "payer_name": d.score_breakdown.payer_name,
            "cpt_weight": d.score_breakdown.cpt_weight,
            "cpt_label": d.score_breakdown.cpt_label,
            "issue_weight": d.score_breakdown.issue_weight,
            "final_score": d.score_breakdown.final_score,
        },
    }
