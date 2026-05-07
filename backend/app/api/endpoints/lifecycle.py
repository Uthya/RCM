"""API endpoints for claim lifecycle tracking and fix effectiveness."""

from fastapi import APIRouter, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from app.api.deps import get_db
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
    session: AsyncSession = Depends(get_db),
):
    docs, total = await get_lifecycles(
        session,
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
async def lifecycle_summary(session: AsyncSession = Depends(get_db)):
    return await get_lifecycle_stats(session)


@router.get("/stats/fix-effectiveness")
async def fix_effectiveness(
    payer_name: str | None = Query(None),
    issue_type: str | None = Query(None),
    session: AsyncSession = Depends(get_db),
):
    return await get_fix_stats(session, payer_name=payer_name, issue_type=issue_type)


@router.get("/{claim_id}")
async def get_claim_lifecycle(claim_id: str, session: AsyncSession = Depends(get_db)):
    doc = await get_lifecycle(session, claim_id)
    if not doc:
        return {"detail": "No lifecycle found for this claim", "claim_id": claim_id}
    return doc
