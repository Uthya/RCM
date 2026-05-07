"""
Decision Engine — single source of truth for claim action routing.

Computes a composite score from ML model output + payer weight + CPT risk
pattern + issue severity, then routes to auto_submit / review / fix_required.

Weights are loaded from PostgreSQL at startup (falls back to defaults).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import config_repo

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
CRITICAL_ISSUE_WEIGHTS: dict[str, float] = {
    "missing_npi": 0.25,
    "missing_modifier": 0.15,
    "missing_prior_auth": 0.10,
    "invalid_cpt": 0.20,
    "missing_payer_id": 0.15,
    "weak_diagnosis": 0.10,
    "vague_sole_diagnosis": 0.15,
    "missing_dob": 0.05,
    "concentration_risk": 0.08,
}

# ── Payer-specific modifier escalation ──
PAYER_MODIFIER_MULTIPLIER: dict[str, float] = {
    "MEDICAID": 2.0,
    "MA": 2.0,
    "MEDICARE": 1.5,
    "TRICARE": 1.5,
}

# ── Non-specific / vague ICD-10 codes ──
VAGUE_DIAGNOSIS_CODES: set[str] = {
    "R69", "Z741", "Z739", "Z762", "Z711", "R688", "R6889", "Z7689",
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
    action: str
    action_label: str
    can_submit: bool
    requires_review: bool
    threshold_note: str
    score_breakdown: ScoreBreakdown


# ── In-memory config (loaded from DB at startup) ──

_payer_weights: dict[str, float] = {}
_cpt_patterns: list[dict] = []
_config_loaded: bool = False


async def load_config(session: AsyncSession) -> None:
    """Load decision config from DB. Falls back to defaults if empty."""
    global _payer_weights, _cpt_patterns, _config_loaded

    try:
        pw = await config_repo.get_payer_weights(session)
        _payer_weights = pw if pw else dict(DEFAULT_PAYER_WEIGHTS)
    except Exception:
        _payer_weights = dict(DEFAULT_PAYER_WEIGHTS)

    try:
        cpt_docs = await config_repo.get_cpt_patterns(session)
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
    global _payer_weights, _cpt_patterns, _config_loaded
    if not _config_loaded:
        _payer_weights = dict(DEFAULT_PAYER_WEIGHTS)
        _cpt_patterns = list(DEFAULT_CPT_RISK_PATTERNS)
        _config_loaded = True


def classify_issues(issues: list[dict], payer_name: str = "") -> list[dict]:
    """Classify validation issues into weighted categories."""
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
    """Compute composite score and return action decision."""
    _ensure_defaults()

    payer_key = payer_name.upper().split()[0] if payer_name else ""
    payer_w = _payer_weights.get(payer_key, 0.0)

    cpt_w = 0.0
    cpt_label = ""
    for pat in sorted(_cpt_patterns, key=lambda p: len(p.get("cpt_prefix", "")), reverse=True):
        prefix = pat.get("cpt_prefix", "")
        if prefix and primary_cpt.startswith(prefix):
            cpt_w = pat.get("weight", 0.0)
            cpt_label = pat.get("label", "")
            break

    issue_details: list[dict] = []
    if issues is not None and len(issues) > 0:
        issue_details = classify_issues(issues, payer_name=payer_name)
        issue_w = sum(d["weight"] for d in issue_details)
    else:
        if issue_count >= 5:
            issue_w = 0.15
        elif issue_count >= 3:
            issue_w = 0.08
        elif issue_count >= 1:
            issue_w = 0.04
        else:
            issue_w = 0.0

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
    """Serialize a ClaimDecision to a plain dict for API responses."""
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
