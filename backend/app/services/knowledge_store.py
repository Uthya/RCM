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


async def record_fix(
    session: AsyncSession,
    claim_id: str,
    issue_type: str,
    fix_applied: str,
    payer_name: str,
    cpt_code: str,
    outcome: str,
) -> None:
    """Record a fix outcome. Writes to both fix_history and fix_effectiveness."""
    await knowledge_repo.insert_fix(session, {
        "claim_id": claim_id,
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
