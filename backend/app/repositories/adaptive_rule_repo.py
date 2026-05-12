"""Repository for adaptive_rules table — CRUD, batch queries, confidence graduation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func, update, and_, or_, case, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AdaptiveRule


def _graduate(total: int, denial_rate: float) -> tuple[str, str]:
    """Compute confidence_level and severity from evidence counts."""
    if total >= 50 and denial_rate >= 0.70:
        return "HIGH", "ERROR"
    if total >= 20 and denial_rate >= 0.50:
        return "MEDIUM", "WARNING"
    return "LOW", "INFO"


async def upsert_rule(
    session: AsyncSession,
    *,
    rule_type: str,
    payer_name: str,
    cpt_code: str = "",
    carc_code: str = "",
    diagnosis_code: str = "",
    total_claims: int,
    denied_claims: int,
    rule_description: str,
    fix_suggestion: str,
    issue_type: str,
    threshold_value: float | None = None,
) -> AdaptiveRule:
    """Insert or update an adaptive rule using ON CONFLICT DO UPDATE.

    Recalculates confidence/severity on every upsert.
    """
    denial_rate = denied_claims / total_claims if total_claims > 0 else 0.0
    confidence, severity = _graduate(total_claims, denial_rate)
    now = datetime.now(timezone.utc)

    stmt = pg_insert(AdaptiveRule).values(
        rule_type=rule_type,
        payer_name=payer_name,
        cpt_code=cpt_code,
        carc_code=carc_code,
        diagnosis_code=diagnosis_code,
        total_claims=total_claims,
        denied_claims=denied_claims,
        denial_rate=round(denial_rate, 4),
        confidence_level=confidence,
        severity=severity,
        rule_description=rule_description,
        fix_suggestion=fix_suggestion,
        issue_type=issue_type,
        threshold_value=threshold_value,
        is_active=True,
        last_mined_at=now,
        updated_at=now,
    )

    stmt = stmt.on_conflict_do_update(
        constraint="uq_adaptive_rules_identity",
        set_={
            "total_claims": total_claims,
            "denied_claims": denied_claims,
            "denial_rate": round(denial_rate, 4),
            "confidence_level": confidence,
            "severity": severity,
            "rule_description": rule_description,
            "fix_suggestion": fix_suggestion,
            "issue_type": issue_type,
            "threshold_value": threshold_value,
            "is_active": True,
            "last_mined_at": now,
            "updated_at": now,
            "retired_at": None,
        },
    )

    await session.execute(stmt)
    await session.flush()

    # Fetch the upserted row
    result = await session.execute(
        select(AdaptiveRule).where(
            AdaptiveRule.rule_type == rule_type,
            AdaptiveRule.payer_name == payer_name,
            AdaptiveRule.cpt_code == cpt_code,
            AdaptiveRule.carc_code == carc_code,
            AdaptiveRule.diagnosis_code == diagnosis_code,
        )
    )
    return result.scalar_one()


async def get_active_rules_batch(
    session: AsyncSession,
    payer_cpt_pairs: list[tuple[str, str]],
    min_confidence: str = "MEDIUM",
) -> dict[tuple[str, str], list[AdaptiveRule]]:
    """Single query for all (payer, CPT) pairs. Returns dict keyed by (payer, cpt).

    min_confidence filters: MEDIUM = MEDIUM+HIGH, HIGH = HIGH only.
    """
    if not payer_cpt_pairs:
        return {}

    confidence_levels = ["HIGH"]
    if min_confidence == "MEDIUM":
        confidence_levels = ["MEDIUM", "HIGH"]
    elif min_confidence == "LOW":
        confidence_levels = ["LOW", "MEDIUM", "HIGH"]

    # Build OR conditions for all pairs, including rules with empty cpt_code (payer-wide)
    conditions = []
    for payer, cpt in payer_cpt_pairs:
        conditions.append(
            and_(
                AdaptiveRule.payer_name == payer,
                or_(AdaptiveRule.cpt_code == cpt, AdaptiveRule.cpt_code == ""),
            )
        )

    stmt = (
        select(AdaptiveRule)
        .where(
            AdaptiveRule.is_active == True,  # noqa: E712
            AdaptiveRule.confidence_level.in_(confidence_levels),
            or_(*conditions),
        )
    )

    result = await session.execute(stmt)
    rules = result.scalars().all()

    # Group by (payer, cpt) — a rule with cpt_code="" matches all CPTs for that payer
    grouped: dict[tuple[str, str], list[AdaptiveRule]] = {}
    payer_set = {p for p, _ in payer_cpt_pairs}
    for rule in rules:
        if rule.cpt_code:
            key = (rule.payer_name, rule.cpt_code)
            grouped.setdefault(key, []).append(rule)
        else:
            # Payer-wide rule — attach to all CPTs for this payer
            for payer, cpt in payer_cpt_pairs:
                if payer == rule.payer_name:
                    grouped.setdefault((payer, cpt), []).append(rule)

    return grouped


async def get_active_rules_for_payer(
    session: AsyncSession,
    payer_name: str,
    cpt_codes: list[str],
    min_confidence: str = "MEDIUM",
) -> list[AdaptiveRule]:
    """Single-claim query: get all active adaptive rules for a payer + CPT codes."""
    confidence_levels = ["HIGH"]
    if min_confidence == "MEDIUM":
        confidence_levels = ["MEDIUM", "HIGH"]
    elif min_confidence == "LOW":
        confidence_levels = ["LOW", "MEDIUM", "HIGH"]

    stmt = (
        select(AdaptiveRule)
        .where(
            AdaptiveRule.is_active == True,  # noqa: E712
            AdaptiveRule.payer_name == payer_name,
            AdaptiveRule.confidence_level.in_(confidence_levels),
            or_(
                AdaptiveRule.cpt_code.in_(cpt_codes),
                AdaptiveRule.cpt_code == "",
            ),
        )
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def retire_stale_rules(
    session: AsyncSession,
    max_age_days: int = 90,
) -> int:
    """Deactivate rules not reinforced for max_age_days or with low denial rate.

    Rules with operator_approved=True are never retired automatically.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    now = datetime.now(timezone.utc)

    stmt = (
        update(AdaptiveRule)
        .where(
            AdaptiveRule.is_active == True,  # noqa: E712
            or_(
                AdaptiveRule.operator_approved == None,  # noqa: E711
                AdaptiveRule.operator_approved != True,  # noqa: E712
            ),
            or_(
                AdaptiveRule.updated_at < cutoff,
                AdaptiveRule.denial_rate < 0.20,
            ),
        )
        .values(
            is_active=False,
            retired_at=now,
            updated_at=now,
        )
    )

    result = await session.execute(stmt)
    await session.flush()
    return result.rowcount


async def get_rules_audit(
    session: AsyncSession,
    payer_name: str | None = None,
    rule_type: str | None = None,
    confidence_level: str | None = None,
    is_active: bool | None = None,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[AdaptiveRule], int]:
    """Paginated view for admin audit API."""
    stmt = select(AdaptiveRule)
    count_stmt = select(func.count()).select_from(AdaptiveRule)

    filters = []
    if payer_name:
        filters.append(AdaptiveRule.payer_name == payer_name)
    if rule_type:
        filters.append(AdaptiveRule.rule_type == rule_type)
    if confidence_level:
        filters.append(AdaptiveRule.confidence_level == confidence_level)
    if is_active is not None:
        filters.append(AdaptiveRule.is_active == is_active)

    if filters:
        stmt = stmt.where(*filters)
        count_stmt = count_stmt.where(*filters)

    count_result = await session.execute(count_stmt)
    total = count_result.scalar() or 0

    stmt = stmt.order_by(AdaptiveRule.updated_at.desc()).offset(skip).limit(limit)
    result = await session.execute(stmt)
    rules = list(result.scalars().all())

    return rules, total


async def get_rules_stats(session: AsyncSession) -> dict:
    """Summary counts by confidence, severity, and rule_type."""
    # Confidence counts
    conf_stmt = (
        select(
            AdaptiveRule.confidence_level,
            func.count().label("count"),
        )
        .where(AdaptiveRule.is_active == True)  # noqa: E712
        .group_by(AdaptiveRule.confidence_level)
    )
    conf_result = await session.execute(conf_stmt)
    by_confidence = {row.confidence_level: row.count for row in conf_result}

    # Severity counts
    sev_stmt = (
        select(
            AdaptiveRule.severity,
            func.count().label("count"),
        )
        .where(AdaptiveRule.is_active == True)  # noqa: E712
        .group_by(AdaptiveRule.severity)
    )
    sev_result = await session.execute(sev_stmt)
    by_severity = {row.severity: row.count for row in sev_result}

    # Rule type counts
    type_stmt = (
        select(
            AdaptiveRule.rule_type,
            func.count().label("count"),
        )
        .where(AdaptiveRule.is_active == True)  # noqa: E712
        .group_by(AdaptiveRule.rule_type)
    )
    type_result = await session.execute(type_stmt)
    by_rule_type = {row.rule_type: row.count for row in type_result}

    # Total active + retired
    total_active = sum(by_confidence.values())
    retired_stmt = select(func.count()).select_from(AdaptiveRule).where(
        AdaptiveRule.is_active == False  # noqa: E712
    )
    retired_result = await session.execute(retired_stmt)
    total_retired = retired_result.scalar() or 0

    return {
        "total_active": total_active,
        "total_retired": total_retired,
        "by_confidence": by_confidence,
        "by_severity": by_severity,
        "by_rule_type": by_rule_type,
    }


async def operator_override(
    session: AsyncSession,
    rule_id: int,
    approved: bool,
    notes: str | None = None,
) -> AdaptiveRule | None:
    """Set operator_approved + operator_notes on a rule."""
    result = await session.execute(
        select(AdaptiveRule).where(AdaptiveRule.id == rule_id)
    )
    rule = result.scalar_one_or_none()
    if not rule:
        return None

    rule.operator_approved = approved
    rule.operator_notes = notes
    rule.updated_at = datetime.now(timezone.utc)

    if not approved:
        rule.is_active = False

    await session.flush()
    return rule
