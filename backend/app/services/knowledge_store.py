"""
Knowledge Layer — Historical fix reuse.

Records which fixes actually worked for specific payer + issue type + CPT
combinations, then surfaces the best-performing fix as a recommendation.

Uses a `fix_history` MongoDB collection with minimum sample guards to
prevent small-sample bias.
"""

from __future__ import annotations

from datetime import datetime

import structlog

from app.db.mongodb import get_db

logger = structlog.get_logger()

MIN_FIX_SAMPLES = 10  # minimum cases before trusting a fix recommendation


async def record_fix(
    claim_id: str,
    issue_type: str,
    fix_applied: str,
    payer_name: str,
    cpt_code: str,
    outcome: str,
) -> None:
    """Record a fix outcome when a claim is resubmitted and the result is known.

    Called automatically when 835 outcome arrives for a claim that had issues.
    """
    db = get_db()
    await db.fix_history.insert_one({
        "claim_id": claim_id,
        "issue_type": issue_type,
        "fix_applied": fix_applied,
        "payer_name": payer_name,
        "cpt_code": cpt_code,
        "outcome": outcome,  # "paid" or "denied"
        "created_at": datetime.utcnow(),
    })
    logger.info(
        "Fix recorded",
        claim_id=claim_id,
        issue_type=issue_type,
        outcome=outcome,
    )


async def get_best_fix(
    issue_type: str,
    payer_name: str,
    cpt_code: str = "",
) -> dict | None:
    """Find the most successful fix for this issue + payer + CPT combination.

    Returns None if insufficient data (< MIN_FIX_SAMPLES) to avoid
    small-sample bias and early incorrect learning.
    """
    db = get_db()

    match_filter: dict = {"issue_type": issue_type, "payer_name": payer_name}
    if cpt_code:
        match_filter["cpt_code"] = cpt_code

    pipeline = [
        {"$match": match_filter},
        {"$group": {
            "_id": "$fix_applied",
            "total": {"$sum": 1},
            "success": {"$sum": {"$cond": [{"$eq": ["$outcome", "paid"]}, 1, 0]}},
        }},
        {"$match": {"total": {"$gte": MIN_FIX_SAMPLES}}},
        {"$addFields": {"success_rate": {"$divide": ["$success", "$total"]}}},
        {"$sort": {"success_rate": -1}},
        {"$limit": 1},
    ]

    result = await db.fix_history.aggregate(pipeline).to_list(1)
    if result and result[0]["success_rate"] > 0.5:
        total = result[0]["total"]
        rate = result[0]["success_rate"]
        return {
            "fix": result[0]["_id"],
            "success_rate": round(rate, 2),
            "sample_size": total,
            "confidence": "high" if total >= 50 else "moderate",
        }
    return None


async def get_top_fixes(
    issue_type: str,
    payer_name: str,
    cpt_code: str = "",
    limit: int = 3,
) -> list[dict]:
    """Return top N fixes ranked by success_rate for a given issue + payer + CPT.

    Same guards as get_best_fix: MIN_FIX_SAMPLES and success_rate > 0.5.
    """
    db = get_db()

    match_filter: dict = {"issue_type": issue_type, "payer_name": payer_name}
    if cpt_code:
        match_filter["cpt_code"] = cpt_code

    pipeline = [
        {"$match": match_filter},
        {"$group": {
            "_id": "$fix_applied",
            "total": {"$sum": 1},
            "success": {"$sum": {"$cond": [{"$eq": ["$outcome", "paid"]}, 1, 0]}},
        }},
        {"$match": {"total": {"$gte": MIN_FIX_SAMPLES}}},
        {"$addFields": {"success_rate": {"$divide": ["$success", "$total"]}}},
        {"$match": {"success_rate": {"$gt": 0.5}}},
        {"$sort": {"success_rate": -1}},
        {"$limit": limit},
    ]

    results = await db.fix_history.aggregate(pipeline).to_list(limit)
    return [
        {
            "fix": r["_id"],
            "success_rate": round(r["success_rate"], 2),
            "sample_size": r["total"],
            "confidence": "high" if r["total"] >= 50 else "moderate",
        }
        for r in results
    ]


async def get_fix_stats(
    payer_name: str | None = None,
    issue_type: str | None = None,
) -> dict:
    """Aggregate fix success rates grouped by payer + CPT + issue_type + fix.

    Separates 'qualified' fixes (>= MIN_FIX_SAMPLES) from 'learning'
    fixes (< MIN_FIX_SAMPLES) so the UI can show what's trusted vs
    still collecting data.
    """
    db = get_db()

    match_filter: dict = {}
    if payer_name:
        match_filter["payer_name"] = payer_name
    if issue_type:
        match_filter["issue_type"] = issue_type

    pipeline: list[dict] = []
    if match_filter:
        pipeline.append({"$match": match_filter})

    pipeline.extend([
        {"$group": {
            "_id": {
                "payer_name": "$payer_name",
                "cpt_code": "$cpt_code",
                "issue_type": "$issue_type",
                "fix_applied": "$fix_applied",
            },
            "total": {"$sum": 1},
            "success": {"$sum": {"$cond": [{"$eq": ["$outcome", "paid"]}, 1, 0]}},
        }},
        {"$addFields": {"success_rate": {"$divide": ["$success", "$total"]}}},
        {"$sort": {"success_rate": -1}},
    ])

    all_results = await db.fix_history.aggregate(pipeline).to_list(500)

    qualified = []
    learning = []
    for r in all_results:
        entry = {
            "payer_name": r["_id"]["payer_name"],
            "cpt_code": r["_id"]["cpt_code"],
            "issue_type": r["_id"]["issue_type"],
            "fix_applied": r["_id"]["fix_applied"],
            "success_rate": round(r["success_rate"], 2),
            "sample_size": r["total"],
        }
        if r["total"] >= MIN_FIX_SAMPLES:
            qualified.append(entry)
        else:
            learning.append(entry)

    return {
        "qualified": qualified,
        "learning": learning,
        "total_fix_records": sum(r["total"] for r in all_results),
    }


async def get_best_fixes_batch(
    issue_types: list[str],
    payer_name: str,
    cpt_codes: list[str] | None = None,
) -> dict[str, dict | None]:
    """Batch lookup of best fixes for multiple issue types."""
    results = {}
    for i, issue_type in enumerate(issue_types):
        cpt = cpt_codes[i] if cpt_codes and i < len(cpt_codes) else ""
        results[issue_type] = await get_best_fix(issue_type, payer_name, cpt)
    return results
