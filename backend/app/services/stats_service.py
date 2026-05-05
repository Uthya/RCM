"""Aggregate statistics for dashboard."""

from app.db.mongodb import get_db
from app.schemas.dashboard import (
    DashboardSummary, RiskDistribution, RiskBucket,
    PayerStatsResponse, PayerStat,
)


async def get_summary() -> DashboardSummary:
    """Get dashboard summary statistics."""
    db = get_db()

    total_claims = await db.claims.count_documents({})
    total_predicted = await db.predictions.count_documents({})
    total_remittances = await db.remittances.count_documents({})

    high_risk = await db.predictions.count_documents({"risk_level": "HIGH"})
    medium_risk = await db.predictions.count_documents({"risk_level": "MEDIUM"})
    low_risk = await db.predictions.count_documents({"risk_level": "LOW"})

    # Denial rate from 835 outcomes
    total_with_outcome = await db.remittances.count_documents({})
    denied_count = await db.remittances.count_documents({"claim_status": "denied"})
    denial_rate = denied_count / total_with_outcome if total_with_outcome > 0 else 0.0

    # Totals
    billed_pipeline = [{"$group": {"_id": None, "total": {"$sum": "$total_charge"}}}]
    billed_result = await db.claims.aggregate(billed_pipeline).to_list(1)
    total_billed = billed_result[0]["total"] if billed_result else 0.0

    paid_pipeline = [{"$group": {"_id": None, "total": {"$sum": "$paid_amount"}}}]
    paid_result = await db.remittances.aggregate(paid_pipeline).to_list(1)
    total_paid = paid_result[0]["total"] if paid_result else 0.0

    return DashboardSummary(
        total_claims=total_claims,
        total_predicted=total_predicted,
        high_risk_count=high_risk,
        medium_risk_count=medium_risk,
        low_risk_count=low_risk,
        total_remittances=total_remittances,
        denial_rate=round(denial_rate, 4),
        total_billed=round(total_billed, 2),
        total_paid=round(total_paid, 2),
    )


async def get_risk_distribution() -> RiskDistribution:
    """Get risk score histogram data."""
    db = get_db()

    pipeline = [
        {
            "$bucket": {
                "groupBy": "$risk_score",
                "boundaries": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
                "default": "other",
                "output": {"count": {"$sum": 1}},
            }
        }
    ]

    results = await db.predictions.aggregate(pipeline).to_list(20)

    buckets = []
    boundaries = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i, low in enumerate(boundaries):
        high = boundaries[i + 1] if i + 1 < len(boundaries) else 1.0
        label = f"{low:.1f}-{high:.1f}"
        count = 0
        for r in results:
            if r["_id"] == low:
                count = r["count"]
                break
        buckets.append(RiskBucket(range_label=label, count=count))

    return RiskDistribution(buckets=buckets)


async def get_payer_stats() -> PayerStatsResponse:
    """Get denial rate by payer from 835 outcomes."""
    db = get_db()

    pipeline = [
        {
            "$group": {
                "_id": {"payer_name": "$payer_name"},
                "total_claims": {"$sum": 1},
                "denied_count": {
                    "$sum": {"$cond": [{"$eq": ["$claim_status", "denied"]}, 1, 0]}
                },
            }
        },
        {"$sort": {"total_claims": -1}},
    ]

    results = await db.remittances.aggregate(pipeline).to_list(100)

    payers = []
    for r in results:
        total = r["total_claims"]
        denied = r["denied_count"]
        payers.append(PayerStat(
            payer_name=r["_id"].get("payer_name", "Unknown"),
            payer_id="",
            total_claims=total,
            denied_count=denied,
            denial_rate=round(denied / total, 4) if total > 0 else 0.0,
        ))

    return PayerStatsResponse(payers=payers)
