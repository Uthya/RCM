"""API endpoints for claim lifecycle tracking and fix effectiveness."""

from fastapi import APIRouter, Query

import structlog

from app.services.lifecycle_service import (
    get_lifecycle,
    get_lifecycles,
    get_lifecycle_stats,
)
from app.services.knowledge_store import get_fix_stats

logger = structlog.get_logger()
router = APIRouter(prefix="/lifecycle", tags=["lifecycle"])


@router.get("")
async def list_lifecycles(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: str | None = Query(None, description="Filter by current_status (PENDING, PAID, DENIED, PARTIAL)"),
    min_attempts: int | None = Query(None, ge=1, description="Minimum number of attempts"),
    payer: str | None = Query(None, description="Filter by payer name (case-insensitive substring)"),
):
    """Paginated list of claim lifecycles with optional filters."""
    docs, total = await get_lifecycles(
        skip=skip,
        limit=limit,
        status=status,
        min_attempts=min_attempts,
        payer_name=payer,
    )
    return {
        "lifecycles": docs,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/stats/summary")
async def lifecycle_summary():
    """Aggregate lifecycle statistics: first-pass payment rate, avg attempts, resubmission success rate."""
    return await get_lifecycle_stats()


@router.get("/stats/fix-effectiveness")
async def fix_effectiveness(
    payer_name: str | None = Query(None),
    issue_type: str | None = Query(None),
):
    """Fix success rates grouped by payer + CPT + issue + fix.

    Returns qualified fixes (>= 10 samples) and learning fixes (< 10 samples).
    """
    return await get_fix_stats(payer_name=payer_name, issue_type=issue_type)


@router.get("/{claim_id}")
async def get_claim_lifecycle(claim_id: str):
    """Full lifecycle for a single claim including all attempts."""
    doc = await get_lifecycle(claim_id)
    if not doc:
        return {"detail": "No lifecycle found for this claim", "claim_id": claim_id}
    return doc
