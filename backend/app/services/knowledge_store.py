"""
Knowledge Layer — Historical fix reuse.

Records which fixes actually worked for specific payer + issue type + CPT
combinations, then surfaces the best-performing fix as a recommendation.

Uses PostgreSQL fix_history and fix_effectiveness tables via repository layer.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import knowledge_repo

logger = structlog.get_logger()

MIN_FIX_SAMPLES = 10

# CARC denial code → fix-tracking issue type
CARC_FIX_MAP: dict[str, str] = {
    "4":   "carc_4_modifier",
    "5":   "carc_5_place_of_service",
    "6":   "carc_6_age_mismatch",
    "9":   "carc_9_dx_age_mismatch",
    "11":  "carc_11_dx_proc_mismatch",
    "16":  "carc_16_submission_error",
    "18":  "carc_18_duplicate",
    "29":  "carc_29_timely_filing",
    "31":  "carc_31_member_id",
    "50":  "carc_50_medical_necessity",
    "96":  "carc_96_non_covered",
    "167": "carc_167_dx_not_covered",
    "197": "carc_197_prior_auth",
    "204": "carc_204_not_covered",
}

# Issue type → relevant change categories for fix attribution filtering.
# None = catch-all (all categories relevant).
ISSUE_RELEVANT_CATEGORIES: dict[str, set[str] | None] = {
    # Validation-based issue types
    "missing_modifier":     {"modifier"},
    "invalid_cpt":          {"cpt", "service_line_count"},
    "missing_prior_auth":   {"prior_auth"},
    "vague_diagnosis":      {"diagnosis"},
    "weak_diagnosis":       {"diagnosis"},
    "missing_npi":          {"npi"},
    "missing_dob":          {"patient_dob"},
    "missing_payer_id":     {"payer_id"},
    # CARC-based issue types
    "carc_4_modifier":           {"modifier"},
    "carc_5_place_of_service":   {"place_of_service"},
    "carc_6_age_mismatch":       {"patient_dob", "diagnosis"},
    "carc_9_dx_age_mismatch":    {"diagnosis", "patient_dob"},
    "carc_11_dx_proc_mismatch":  {"diagnosis", "cpt"},
    "carc_16_submission_error":  None,  # catch-all
    "carc_18_duplicate":         {"frequency_code", "service_date", "other"},
    "carc_29_timely_filing":     {"service_date", "frequency_code"},
    "carc_31_member_id":         {"subscriber_id"},
    "carc_50_medical_necessity": {"diagnosis", "modifier"},
    "carc_96_non_covered":       {"diagnosis", "cpt", "modifier"},
    "carc_167_dx_not_covered":   {"diagnosis"},
    "carc_197_prior_auth":       {"prior_auth"},
    "carc_204_not_covered":      {"cpt", "diagnosis"},
}


def filter_changes_for_issue(
    categorized_changes: dict[str, list[str]],
    issue_type: str,
) -> str:
    """Return semicolon-joined string of only changes relevant to this issue type."""
    if not categorized_changes:
        return ""
    # Unknown issue types or explicit catch-all → return everything
    if issue_type not in ISSUE_RELEVANT_CATEGORIES:
        return categorized_changes.get("_flat", "")
    relevant_cats = ISSUE_RELEVANT_CATEGORIES[issue_type]
    if relevant_cats is None:
        return categorized_changes.get("_flat", "")
    parts: list[str] = []
    for cat in sorted(relevant_cats):
        parts.extend(categorized_changes.get(cat, []))
    return "; ".join(parts)


async def record_fix(
    session: AsyncSession,
    claim_id: str,
    issue_type: str,
    fix_applied: str,
    payer_name: str,
    cpt_code: str,
    outcome: str,
    attempt_number: int | None = None,
) -> None:
    """Record a fix outcome. Writes to both fix_history and fix_effectiveness."""
    await knowledge_repo.insert_fix(session, {
        "claim_id": claim_id,
        "attempt_number": attempt_number,
        "issue_type": issue_type,
        "fix_applied": fix_applied,
        "payer_name": payer_name,
        "cpt_code": cpt_code,
        "outcome": outcome,
        "created_at": datetime.utcnow(),
    })

    await knowledge_repo.upsert_effectiveness(session, {
        "payer_name": payer_name,
        "cpt_code": cpt_code,
        "issue_type": issue_type,
        "fix_applied": fix_applied,
        "outcome": outcome,
    })

    logger.info("Fix recorded", claim_id=claim_id, issue_type=issue_type, outcome=outcome)


async def record_carc_fix(
    session: AsyncSession,
    claim_id: str,
    carc_codes: list[str],
    fix_applied: str,
    payer_name: str,
    cpt_code: str,
    outcome: str,
    attempt_number: int | None = None,
) -> None:
    """Map CARC denial codes to issue types and record each as a fix outcome."""
    for code in carc_codes:
        issue_type = CARC_FIX_MAP.get(code)
        if not issue_type:
            continue
        await record_fix(
            session,
            claim_id=claim_id,
            issue_type=issue_type,
            fix_applied=fix_applied,
            payer_name=payer_name,
            cpt_code=cpt_code,
            outcome=outcome,
            attempt_number=attempt_number,
        )


async def get_best_fix(
    session: AsyncSession,
    issue_type: str,
    payer_name: str,
    cpt_code: str = "",
) -> dict | None:
    return await knowledge_repo.get_best_fix(
        session, issue_type, payer_name, cpt_code, min_samples=MIN_FIX_SAMPLES,
    )


async def get_top_fixes(
    session: AsyncSession,
    issue_type: str,
    payer_name: str,
    cpt_code: str = "",
    limit: int = 3,
) -> list[dict]:
    return await knowledge_repo.get_top_fixes(
        session, issue_type, payer_name, cpt_code, limit=limit, min_samples=MIN_FIX_SAMPLES,
    )


async def get_fix_stats(
    session: AsyncSession,
    payer_name: str | None = None,
    issue_type: str | None = None,
) -> dict:
    return await knowledge_repo.get_fix_stats(
        session, payer_name=payer_name, issue_type=issue_type, min_samples=MIN_FIX_SAMPLES,
    )


async def get_best_fixes_batch(
    session: AsyncSession,
    issue_types: list[str],
    payer_name: str,
    cpt_codes: list[str] | None = None,
) -> dict[str, dict | None]:
    results = {}
    for i, issue_type in enumerate(issue_types):
        cpt = cpt_codes[i] if cpt_codes and i < len(cpt_codes) else ""
        results[issue_type] = await get_best_fix(session, issue_type, payer_name, cpt)
    return results
