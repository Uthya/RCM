"""
Rule Miner — discovers adaptive validation rules from denial patterns.

Three mining strategies (CARC-driven, diagnosis-denial, charge threshold)
plus two Tier-1 config updaters (payer weights, CPT risk patterns).
"""

from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import adaptive_rule_repo

logger = structlog.get_logger()

# CARC code → (rule_type, issue_type, description_template)
CARC_RULE_MAP: dict[str, tuple[str, str, str]] = {
    "4":   ("carc_modifier",         "missing_modifier",     "Modifier required"),
    "11":  ("carc_dx_mismatch",      "dx_cpt_mismatch",      "Diagnosis/procedure mismatch"),
    "29":  ("carc_timely_filing",    "timely_filing",         "Timely filing limit exceeded"),
    "31":  ("carc_member_id",        "invalid_member_id",     "Invalid subscriber/member ID"),
    "50":  ("carc_medical_necessity","medical_necessity",     "Medical necessity not established"),
    "167": ("carc_dx_not_covered",   "dx_not_covered",        "Diagnosis not covered by plan"),
    "197": ("carc_prior_auth",       "missing_prior_auth",    "Prior authorization required"),
}

# Generic fix suggestions by CARC code
CARC_GENERIC_FIXES: dict[str, str] = {
    "4":   "Review modifier requirements for this CPT code with the payer",
    "11":  "Verify diagnosis codes support medical necessity for this procedure",
    "29":  "Ensure claims are submitted within payer's timely filing deadline",
    "31":  "Verify subscriber/member ID against the insurance card before submission",
    "50":  "Strengthen medical necessity documentation; add supporting diagnosis codes",
    "167": "Confirm diagnosis code is covered under the patient's plan before billing",
    "197": "Obtain prior authorization from the payer before submitting",
}


async def mine_carc_rules(session: AsyncSession) -> int:
    """Mine CARC-driven rules from remittance data.

    For each (payer, CPT, CARC) triple with >= 10 total claims and >= 5 CARC
    occurrences at >= 40% denial rate, upsert an adaptive rule.
    """
    sql = text("""
        WITH remittance_cpt AS (
            SELECT
                r.payer_name,
                sl->>'cpt_code' AS cpt_code,
                r.claim_status,
                r.carc_codes,
                r.claim_id
            FROM remittances r,
                 jsonb_array_elements(r.service_lines) sl
            WHERE r.payer_name IS NOT NULL
              AND sl->>'cpt_code' IS NOT NULL
              AND sl->>'cpt_code' != ''
        ),
        payer_cpt_totals AS (
            SELECT
                payer_name,
                cpt_code,
                COUNT(DISTINCT claim_id) AS total_claims
            FROM remittance_cpt
            GROUP BY payer_name, cpt_code
            HAVING COUNT(DISTINCT claim_id) >= 10
        ),
        carc_counts AS (
            SELECT
                rc.payer_name,
                rc.cpt_code,
                carc.code AS carc_code,
                COUNT(DISTINCT rc.claim_id) AS carc_count,
                COUNT(DISTINCT rc.claim_id) FILTER (
                    WHERE rc.claim_status = 'denied'
                ) AS denied_count
            FROM remittance_cpt rc,
                 unnest(rc.carc_codes) AS carc(code)
            GROUP BY rc.payer_name, rc.cpt_code, carc.code
            HAVING COUNT(DISTINCT rc.claim_id) >= 5
        )
        SELECT
            cc.payer_name,
            cc.cpt_code,
            cc.carc_code,
            pct.total_claims,
            cc.denied_count,
            ROUND(cc.denied_count::numeric / NULLIF(pct.total_claims, 0), 4) AS denial_rate
        FROM carc_counts cc
        JOIN payer_cpt_totals pct
          ON cc.payer_name = pct.payer_name
         AND cc.cpt_code = pct.cpt_code
        WHERE cc.denied_count::numeric / NULLIF(pct.total_claims, 0) >= 0.40
    """)

    result = await session.execute(sql)
    rows = result.all()

    upserted = 0
    for row in rows:
        carc = str(row.carc_code).strip()
        mapping = CARC_RULE_MAP.get(carc)
        if not mapping:
            continue

        rule_type, issue_type, desc_template = mapping
        payer = row.payer_name
        cpt = row.cpt_code

        # Try to get a known-good fix from fix_effectiveness
        fix_suggestion = await _get_best_fix_suggestion(session, payer, cpt, issue_type, carc)

        description = (
            f"{payer} frequently denies CPT {cpt} with CARC {carc} ({desc_template}). "
            f"Denial rate: {int(float(row.denial_rate) * 100)}% across {row.total_claims} claims."
        )

        await adaptive_rule_repo.upsert_rule(
            session,
            rule_type=rule_type,
            payer_name=payer,
            cpt_code=cpt,
            carc_code=carc,
            total_claims=row.total_claims,
            denied_claims=row.denied_count,
            rule_description=description,
            fix_suggestion=fix_suggestion,
            issue_type=issue_type,
        )
        upserted += 1

    logger.info("CARC rule mining complete", rules_upserted=upserted)
    return upserted


async def mine_dx_denial_rules(session: AsyncSession) -> int:
    """Mine diagnosis-denial rules.

    When a specific ICD-10 code consistently gets denied with a specific CPT
    for a payer (>= 10 claims, >= 40% denial rate), upsert a dx_cpt_denial rule.
    """
    sql = text("""
        WITH claim_dx AS (
            SELECT
                c.claim_id,
                c.payer_name,
                unnest(c.diagnosis_codes) AS dx_code,
                sl->>'cpt_code' AS cpt_code
            FROM claims c,
                 jsonb_array_elements(c.service_lines) sl
            WHERE c.payer_name IS NOT NULL
              AND c.diagnosis_codes IS NOT NULL
              AND array_length(c.diagnosis_codes, 1) > 0
        ),
        dx_outcomes AS (
            SELECT
                cd.payer_name,
                cd.cpt_code,
                UPPER(REPLACE(cd.dx_code, '.', '')) AS dx_code,
                r.claim_status
            FROM claim_dx cd
            JOIN remittances r ON r.claim_id = cd.claim_id
            WHERE cd.cpt_code IS NOT NULL AND cd.cpt_code != ''
        )
        SELECT
            payer_name,
            cpt_code,
            dx_code,
            COUNT(*) AS total_claims,
            COUNT(*) FILTER (WHERE claim_status = 'denied') AS denied_claims,
            ROUND(
                COUNT(*) FILTER (WHERE claim_status = 'denied')::numeric
                / NULLIF(COUNT(*), 0), 4
            ) AS denial_rate
        FROM dx_outcomes
        GROUP BY payer_name, cpt_code, dx_code
        HAVING COUNT(*) >= 10
           AND COUNT(*) FILTER (WHERE claim_status = 'denied')::numeric
               / NULLIF(COUNT(*), 0) >= 0.40
    """)

    result = await session.execute(sql)
    rows = result.all()

    upserted = 0
    for row in rows:
        description = (
            f"{row.payer_name} frequently denies CPT {row.cpt_code} "
            f"when paired with diagnosis {row.dx_code}. "
            f"Denial rate: {int(float(row.denial_rate) * 100)}% across {row.total_claims} claims."
        )
        fix_suggestion = (
            f"Review whether diagnosis {row.dx_code} adequately supports "
            f"medical necessity for CPT {row.cpt_code} with {row.payer_name}. "
            f"Consider using a more specific diagnosis code."
        )

        await adaptive_rule_repo.upsert_rule(
            session,
            rule_type="dx_cpt_denial",
            payer_name=row.payer_name,
            cpt_code=row.cpt_code,
            diagnosis_code=row.dx_code,
            total_claims=row.total_claims,
            denied_claims=row.denied_claims,
            rule_description=description,
            fix_suggestion=fix_suggestion,
            issue_type="dx_cpt_mismatch",
        )
        upserted += 1

    logger.info("Dx-denial rule mining complete", rules_upserted=upserted)
    return upserted


async def mine_charge_threshold_rules(session: AsyncSession) -> int:
    """Mine charge threshold rules.

    For each (payer, CPT), compute P75 charge amount. If claims above P75
    have >= 50% denial rate with >= 5 samples, upsert a charge_threshold rule.
    """
    sql = text("""
        WITH claim_charges AS (
            SELECT
                c.payer_name,
                sl->>'cpt_code' AS cpt_code,
                (sl->>'charge_amount')::numeric AS charge_amount,
                r.claim_status
            FROM claims c
            JOIN remittances r ON r.claim_id = c.claim_id,
                 jsonb_array_elements(c.service_lines) sl
            WHERE c.payer_name IS NOT NULL
              AND sl->>'cpt_code' IS NOT NULL
              AND sl->>'charge_amount' IS NOT NULL
              AND (sl->>'charge_amount')::numeric > 0
        ),
        percentiles AS (
            SELECT
                payer_name,
                cpt_code,
                percentile_cont(0.75) WITHIN GROUP (ORDER BY charge_amount) AS p75,
                COUNT(*) AS total_claims
            FROM claim_charges
            GROUP BY payer_name, cpt_code
            HAVING COUNT(*) >= 10
        ),
        above_p75 AS (
            SELECT
                cc.payer_name,
                cc.cpt_code,
                p.p75,
                p.total_claims,
                COUNT(*) AS above_count,
                COUNT(*) FILTER (WHERE cc.claim_status = 'denied') AS above_denied
            FROM claim_charges cc
            JOIN percentiles p
              ON cc.payer_name = p.payer_name
             AND cc.cpt_code = p.cpt_code
            WHERE cc.charge_amount > p.p75
            GROUP BY cc.payer_name, cc.cpt_code, p.p75, p.total_claims
            HAVING COUNT(*) >= 5
               AND COUNT(*) FILTER (WHERE cc.claim_status = 'denied')::numeric
                   / NULLIF(COUNT(*), 0) >= 0.50
        )
        SELECT
            payer_name,
            cpt_code,
            p75,
            total_claims,
            above_count,
            above_denied,
            ROUND(above_denied::numeric / NULLIF(above_count, 0), 4) AS denial_rate
        FROM above_p75
    """)

    result = await session.execute(sql)
    rows = result.all()

    upserted = 0
    for row in rows:
        p75 = float(row.p75)
        description = (
            f"{row.payer_name} denies CPT {row.cpt_code} at higher rates when "
            f"charge exceeds ${p75:,.2f} (75th percentile). "
            f"{int(float(row.denial_rate) * 100)}% denial rate above threshold "
            f"({row.above_denied}/{row.above_count} claims)."
        )
        fix_suggestion = (
            f"Review charge amount for CPT {row.cpt_code}. "
            f"Charges above ${p75:,.2f} have significantly higher denial rates with {row.payer_name}. "
            f"Consider adjusting fee schedule or adding supporting documentation."
        )

        await adaptive_rule_repo.upsert_rule(
            session,
            rule_type="charge_threshold",
            payer_name=row.payer_name,
            cpt_code=row.cpt_code,
            total_claims=row.above_count,
            denied_claims=row.above_denied,
            rule_description=description,
            fix_suggestion=fix_suggestion,
            issue_type="charge_threshold",
            threshold_value=p75,
        )
        upserted += 1

    logger.info("Charge threshold rule mining complete", rules_upserted=upserted)
    return upserted


async def update_payer_weights(session: AsyncSession) -> int:
    """Update decision_config payer weights from remittance denial rates.

    Requires >= 20 claims per payer. Scales to weight: min(0.01 + rate * 0.15, 0.20).
    """
    from app.db.models import DecisionConfig
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    sql = text("""
        SELECT
            payer_name,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE claim_status = 'denied') AS denied,
            ROUND(
                COUNT(*) FILTER (WHERE claim_status = 'denied')::numeric
                / NULLIF(COUNT(*), 0), 4
            ) AS denial_rate
        FROM remittances
        WHERE payer_name IS NOT NULL
        GROUP BY payer_name
        HAVING COUNT(*) >= 20
    """)

    result = await session.execute(sql)
    rows = result.all()

    if not rows:
        return 0

    weights = {}
    for row in rows:
        rate = float(row.denial_rate)
        payer_key = row.payer_name.upper().split()[0] if row.payer_name else ""
        if payer_key:
            weights[payer_key] = round(min(0.01 + rate * 0.15, 0.20), 4)

    if not weights:
        return 0

    # Upsert into decision_config
    stmt = pg_insert(DecisionConfig).values(
        type="payer_weights",
        weights=weights,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["type"],
        set_={"weights": weights},
    )
    await session.execute(stmt)
    await session.flush()

    # Hot-reload decision engine
    try:
        from app.services.decision_engine import load_config
        await load_config(session)
    except Exception as e:
        logger.warning("Decision engine reload failed", error=str(e))

    logger.info("Payer weights updated", payer_count=len(weights))
    return len(weights)


async def update_cpt_risk_patterns(session: AsyncSession) -> int:
    """Update cpt_risk_config from CPT prefix denial rates.

    Requires >= 10 claims per prefix, >= 15% denial rate.
    Scales to weight: min(rate * 0.10, 0.05).
    """
    from app.db.models import CptRiskConfig
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    sql = text("""
        WITH cpt_data AS (
            SELECT
                sl->>'cpt_code' AS cpt_code,
                r.claim_status
            FROM remittances r,
                 jsonb_array_elements(r.service_lines) sl
            WHERE sl->>'cpt_code' IS NOT NULL
              AND sl->>'cpt_code' != ''
        )
        SELECT
            LEFT(cpt_code, 3) AS cpt_prefix,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE claim_status = 'denied') AS denied,
            ROUND(
                COUNT(*) FILTER (WHERE claim_status = 'denied')::numeric
                / NULLIF(COUNT(*), 0), 4
            ) AS denial_rate
        FROM cpt_data
        GROUP BY LEFT(cpt_code, 3)
        HAVING COUNT(*) >= 10
           AND COUNT(*) FILTER (WHERE claim_status = 'denied')::numeric
               / NULLIF(COUNT(*), 0) >= 0.15
    """)

    result = await session.execute(sql)
    rows = result.all()

    upserted = 0
    for row in rows:
        rate = float(row.denial_rate)
        weight = round(min(rate * 0.10, 0.05), 4)

        stmt = pg_insert(CptRiskConfig).values(
            cpt_prefix=row.cpt_prefix,
            weight=weight,
            label=f"Prefix {row.cpt_prefix}",
            reason=f"Mined: {int(rate * 100)}% denial rate across {row.total} claims",
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_cpt_risk_config_prefix",
            set_={
                "weight": weight,
                "reason": f"Mined: {int(rate * 100)}% denial rate across {row.total} claims",
            },
        )
        await session.execute(stmt)
        upserted += 1

    await session.flush()

    # Hot-reload decision engine
    try:
        from app.services.decision_engine import load_config
        await load_config(session)
    except Exception as e:
        logger.warning("Decision engine reload failed", error=str(e))

    logger.info("CPT risk patterns updated", patterns_upserted=upserted)
    return upserted


async def run_all_miners(session: AsyncSession) -> dict:
    """Orchestrate all mining strategies + retirement."""
    carc = await mine_carc_rules(session)
    dx = await mine_dx_denial_rules(session)
    charge = await mine_charge_threshold_rules(session)
    retired = await adaptive_rule_repo.retire_stale_rules(session)

    logger.info(
        "Rule mining complete",
        carc_rules=carc,
        dx_rules=dx,
        charge_rules=charge,
        retired=retired,
    )
    return {
        "carc_rules": carc,
        "dx_rules": dx,
        "charge_rules": charge,
        "retired": retired,
    }


async def _get_best_fix_suggestion(
    session: AsyncSession,
    payer: str,
    cpt: str,
    issue_type: str,
    carc_code: str,
) -> str:
    """Try to get a proven fix from fix_effectiveness, fall back to generic."""
    try:
        from app.services.knowledge_store import get_best_fix
        best = await get_best_fix(session, issue_type, payer, cpt)
        if best and best.get("success_rate", 0) > 0.6 and best.get("confidence") == "HIGH":
            rate_pct = int(best["success_rate"] * 100)
            return f"Proven fix ({rate_pct}% success rate): {best['fix']}"
    except Exception:
        pass

    # Also check CARC-specific fix mapping
    carc_issue = f"carc_{carc_code}_" if carc_code else ""
    if carc_issue:
        try:
            from app.services.knowledge_store import get_best_fix
            from app.services.knowledge_store import CARC_FIX_MAP
            carc_issue_type = CARC_FIX_MAP.get(carc_code)
            if carc_issue_type:
                best = await get_best_fix(session, carc_issue_type, payer, cpt)
                if best and best.get("success_rate", 0) > 0.6:
                    rate_pct = int(best["success_rate"] * 100)
                    return f"Proven fix ({rate_pct}% success rate): {best['fix']}"
        except Exception:
            pass

    return CARC_GENERIC_FIXES.get(carc_code, "Review claim details against payer-specific guidelines")
