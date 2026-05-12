"""Admin API for adaptive rules — audit, override, and manual mining trigger."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.repositories import adaptive_rule_repo

router = APIRouter(prefix="/adaptive-rules", tags=["adaptive-rules"])


class OverrideRequest(BaseModel):
    notes: str | None = None


def _rule_to_dict(rule) -> dict:
    return {
        "id": rule.id,
        "rule_type": rule.rule_type,
        "payer_name": rule.payer_name,
        "cpt_code": rule.cpt_code,
        "carc_code": rule.carc_code,
        "diagnosis_code": rule.diagnosis_code,
        "rule_description": rule.rule_description,
        "fix_suggestion": rule.fix_suggestion,
        "issue_type": rule.issue_type,
        "total_claims": rule.total_claims,
        "denied_claims": rule.denied_claims,
        "denial_rate": rule.denial_rate,
        "confidence_level": rule.confidence_level,
        "severity": rule.severity,
        "is_active": rule.is_active,
        "threshold_value": rule.threshold_value,
        "operator_approved": rule.operator_approved,
        "operator_notes": rule.operator_notes,
        "created_at": rule.created_at.isoformat() if rule.created_at else None,
        "updated_at": rule.updated_at.isoformat() if rule.updated_at else None,
        "retired_at": rule.retired_at.isoformat() if rule.retired_at else None,
    }


@router.get("/")
async def list_rules(
    payer_name: str | None = Query(None),
    rule_type: str | None = Query(None),
    confidence: str | None = Query(None),
    active: bool | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
):
    """List adaptive rules with optional filters."""
    rules, total = await adaptive_rule_repo.get_rules_audit(
        session,
        payer_name=payer_name,
        rule_type=rule_type,
        confidence_level=confidence,
        is_active=active,
        skip=skip,
        limit=limit,
    )
    return {
        "rules": [_rule_to_dict(r) for r in rules],
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/stats")
async def rules_stats(session: AsyncSession = Depends(get_db)):
    """Summary counts by confidence, severity, and rule_type."""
    return await adaptive_rule_repo.get_rules_stats(session)


@router.post("/{rule_id}/approve")
async def approve_rule(
    rule_id: int,
    body: OverrideRequest = OverrideRequest(),
    session: AsyncSession = Depends(get_db),
):
    """Operator locks a rule as active (survives retirement)."""
    rule = await adaptive_rule_repo.operator_override(
        session, rule_id, approved=True, notes=body.notes,
    )
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    await session.commit()
    return _rule_to_dict(rule)


@router.post("/{rule_id}/reject")
async def reject_rule(
    rule_id: int,
    body: OverrideRequest = OverrideRequest(),
    session: AsyncSession = Depends(get_db),
):
    """Operator deactivates a rule."""
    rule = await adaptive_rule_repo.operator_override(
        session, rule_id, approved=False, notes=body.notes,
    )
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    await session.commit()
    return _rule_to_dict(rule)


@router.post("/mine-now")
async def mine_now(session: AsyncSession = Depends(get_db)):
    """Manually trigger rule mining."""
    from app.services.rule_miner import run_all_miners, update_payer_weights, update_cpt_risk_patterns

    payer_count = await update_payer_weights(session)
    cpt_count = await update_cpt_risk_patterns(session)
    mining_result = await run_all_miners(session)
    await session.commit()

    return {
        "payer_weights_updated": payer_count,
        "cpt_patterns_updated": cpt_count,
        **mining_result,
    }
