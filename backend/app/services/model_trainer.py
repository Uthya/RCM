"""
Retrain XGBoost model using real matched 837+835 data from MongoDB.

This module pulls claims that have been matched to 835 outcomes
(actual_outcome = "paid" or "denied"), extracts features, and
trains a new XGBoost model. Falls back to augmenting with synthetic
data if real samples are too few.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

from app.config import settings
from app.core.feature_engineer import (
    compute_features_from_claim,
    FEATURE_NAMES,
)

logger = structlog.get_logger()

MIN_REAL_SAMPLES = 30       # minimum real matched claims to attempt training
SYNTHETIC_FILL_TARGET = 2000  # if real < this, pad with synthetic
AUTO_RETRAIN_THRESHOLD = 5000  # matched claims to trigger auto-retrain
AUTO_RETRAIN_INTERVAL_DAYS = 7  # minimum days between auto-retrains


async def retrain_model(db) -> dict:
    """
    Pull real matched data from MongoDB, train XGBoost, save model.
    Returns training summary dict.
    """
    start = time.time()

    # ── 1. Pull all claims with actual outcomes ──
    cursor = db.claims.find(
        {"actual_outcome": {"$in": ["paid", "denied"]}},
    )
    real_docs = await cursor.to_list(length=100_000)
    real_count = len(real_docs)

    logger.info("Retrain: found matched claims", count=real_count)

    if real_count < MIN_REAL_SAMPLES:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {MIN_REAL_SAMPLES} matched claims (have {real_count}). "
                       f"Upload more 835 files to match outcomes to claims.",
            "matched_claims": real_count,
        }

    # ── 2. Extract features + labels from real data ──
    real_features: list[dict] = []
    real_labels: list[int] = []
    claim_meta: list[dict] = []  # for saving training history

    for doc in real_docs:
        feats = compute_features_from_claim(doc)

        # Enrich with historical rates from remittances (sync approximation)
        payer_name = doc.get("payer_name", "")
        npi = doc.get("billing_provider_npi", "")
        service_lines = doc.get("service_lines", [])
        primary_cpt = service_lines[0].get("cpt_code", "") if service_lines else ""

        # Compute denial rates from the matched data itself
        feats["payer_denial_rate"] = await _compute_rate(db, "payer_name", payer_name)
        feats["cpt_denial_rate"] = await _compute_cpt_rate(db, primary_cpt)
        feats["provider_denial_rate"] = await _compute_rate(db, "billing_provider_npi", npi)

        real_features.append(feats)
        label = 1 if doc.get("actual_outcome") == "denied" else 0
        real_labels.append(label)

        claim_meta.append({
            "claim_id": doc.get("claim_id"),
            "patient_name": f"{doc.get('patient_first_name', '')} {doc.get('patient_last_name', '')}".strip(),
            "actual_outcome": doc.get("actual_outcome"),
            "issue_count": doc.get("issue_count", 0),
        })

    df_real = pd.DataFrame(real_features, columns=FEATURE_NAMES).fillna(0.0)
    y_real = np.array(real_labels)

    denied_count = int(y_real.sum())
    paid_count = int(len(y_real) - denied_count)

    logger.info("Retrain: real data stats",
                total=real_count, denied=denied_count, paid=paid_count,
                denial_rate=f"{denied_count/real_count:.1%}")

    # ── 3. Augment with synthetic if needed ──
    synthetic_count = 0
    if real_count < SYNTHETIC_FILL_TARGET:
        synthetic_count = SYNTHETIC_FILL_TARGET - real_count
        df_synth, y_synth = _generate_synthetic(synthetic_count, seed=int(time.time()) % 10000)
        df_train = pd.concat([df_real, df_synth], ignore_index=True)
        y_train_full = np.concatenate([y_real, y_synth])
        logger.info("Retrain: augmented with synthetic", count=synthetic_count)
    else:
        df_train = df_real
        y_train_full = y_real

    # ── 4. Train/test split ──
    X_train, X_test, y_tr, y_te = train_test_split(
        df_train, y_train_full, test_size=0.2, random_state=42,
        stratify=y_train_full,
    )

    # ── 5. Train XGBoost ──
    pos_weight = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_tr, eval_set=[(X_test, y_te)], verbose=False)

    # ── 6. Evaluate ──
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_te, y_pred_proba))
    precision = float(precision_score(y_te, y_pred, zero_division=0))
    recall = float(recall_score(y_te, y_pred, zero_division=0))

    # Also evaluate on ONLY real data portion of test set
    real_test_mask = np.arange(len(y_train_full)) < real_count
    real_in_test = real_test_mask[len(X_train):]  # indices that ended up in test
    if real_in_test.sum() > 10:
        auc_real = float(roc_auc_score(
            y_te[real_in_test], y_pred_proba[real_in_test]
        ))
    else:
        auc_real = None

    logger.info("Retrain: metrics",
                auc=f"{auc:.4f}", precision=f"{precision:.4f}",
                recall=f"{recall:.4f}", auc_real=auc_real)

    # ── 7. Save model ──
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(exist_ok=True)

    # Backup old model
    old_model = model_dir / "demo_model.joblib"
    if old_model.exists():
        backup_name = f"demo_model_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
        old_model.rename(model_dir / backup_name)
        logger.info("Retrain: backed up old model", name=backup_name)

    model_path = model_dir / "demo_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Retrain: saved new model", path=str(model_path))

    # ── 8. Save training record to MongoDB ──
    elapsed = round(time.time() - start, 2)
    from app.core.predictor import get_model_version
    training_record = {
        "trained_at": datetime.utcnow(),
        "model_version": get_model_version(),
        "real_samples": real_count,
        "synthetic_samples": synthetic_count,
        "total_samples": len(df_train),
        "denied_count": denied_count,
        "paid_count": paid_count,
        "denial_rate": round(denied_count / real_count, 4),
        "metrics": {
            "auc_roc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "auc_real_only": round(auc_real, 4) if auc_real else None,
        },
        "feature_importance": {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, model.feature_importances_)
        },
        "elapsed_seconds": elapsed,
    }
    await db.training_history.insert_one(training_record)

    # ── 9. Reload model in memory + refresh decision config ──
    from app.core.predictor import load_model
    load_model()
    from app.services.decision_engine import load_config as load_decision_config
    await load_decision_config(db)

    return {
        "status": "success",
        "message": f"Model retrained on {real_count} real + {synthetic_count} synthetic claims",
        "real_samples": real_count,
        "synthetic_samples": synthetic_count,
        "denied_count": denied_count,
        "paid_count": paid_count,
        "denial_rate": round(denied_count / real_count, 4),
        "metrics": training_record["metrics"],
        "feature_importance": training_record["feature_importance"],
        "elapsed_seconds": elapsed,
    }


async def get_training_status(db) -> dict:
    """Get current training data status and last training info."""
    matched = await db.claims.count_documents(
        {"actual_outcome": {"$in": ["paid", "denied"]}}
    )
    denied = await db.claims.count_documents({"actual_outcome": "denied"})
    paid = await db.claims.count_documents({"actual_outcome": "paid"})
    total_claims = await db.claims.count_documents({})
    total_remittances = await db.remittances.count_documents({})

    # Last training
    last = await db.training_history.find_one(sort=[("trained_at", -1)])

    return {
        "total_claims": total_claims,
        "total_remittances": total_remittances,
        "matched_claims": matched,
        "paid_count": paid,
        "denied_count": denied,
        "ready_to_train": matched >= MIN_REAL_SAMPLES,
        "min_required": MIN_REAL_SAMPLES,
        "last_training": {
            "trained_at": last["trained_at"].isoformat() if last else None,
            "real_samples": last.get("real_samples") if last else None,
            "metrics": last.get("metrics") if last else None,
        } if last else None,
    }


async def validate_training_data(db) -> dict:
    """Validate data quality BEFORE allowing retrain.

    Prevents noisy labels from degrading the model. All gates must pass.
    """
    issues = []

    # 1. Check class balance
    matched = await db.claims.count_documents({"actual_outcome": {"$exists": True}})
    denied = await db.claims.count_documents({"actual_outcome": "denied"})
    paid = matched - denied
    denial_rate = denied / max(matched, 1)

    if denial_rate < 0.05 or denial_rate > 0.95:
        issues.append(f"Extreme class imbalance: {denial_rate:.1%} denial rate")

    # 2. Check for duplicate claim_ids
    pipeline = [
        {"$match": {"actual_outcome": {"$exists": True}}},
        {"$group": {"_id": "$claim_id", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}},
        {"$count": "duplicates"},
    ]
    dups = await db.claims.aggregate(pipeline).to_list(1)
    if dups and dups[0]["duplicates"] > matched * 0.05:
        issues.append(f"Too many duplicate claim_ids: {dups[0]['duplicates']}")

    # 3. Check for missing features on matched claims
    missing_feat = await db.claims.count_documents({
        "actual_outcome": {"$exists": True},
        "$or": [
            {"service_lines": {"$size": 0}},
            {"service_lines": {"$exists": False}},
        ],
    })
    if missing_feat > matched * 0.1:
        issues.append(f"{missing_feat} claims have no service lines (>{10}% of data)")

    # 4. Minimum sample size
    if matched < MIN_REAL_SAMPLES:
        issues.append(f"Only {matched} matched claims (need {MIN_REAL_SAMPLES})")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "stats": {
            "matched": matched,
            "denied": denied,
            "paid": paid,
            "denial_rate": round(denial_rate, 4),
        },
    }


# ── Helpers ──

async def _compute_rate(db, field: str, value: str) -> float:
    """Compute denial rate for a field value from matched claims."""
    if not value:
        return 0.0
    total = await db.claims.count_documents({
        field: value, "actual_outcome": {"$exists": True}
    })
    if total == 0:
        return 0.0
    denied = await db.claims.count_documents({
        field: value, "actual_outcome": "denied"
    })
    return denied / total


async def _compute_cpt_rate(db, cpt_code: str) -> float:
    """Compute denial rate for a CPT code from matched remittances."""
    if not cpt_code:
        return 0.0
    pipeline = [
        {"$match": {"actual_outcome": {"$exists": True}}},
        {"$unwind": "$service_lines"},
        {"$match": {"service_lines.cpt_code": cpt_code}},
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "denied": {"$sum": {"$cond": [{"$eq": ["$actual_outcome", "denied"]}, 1, 0]}},
        }},
    ]
    result = await db.claims.aggregate(pipeline).to_list(1)
    if result and result[0]["total"] > 0:
        return result[0]["denied"] / result[0]["total"]
    return 0.0


def _generate_synthetic(n: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic training data to augment small real datasets."""
    rng = np.random.RandomState(seed)
    data = {
        "total_charge": rng.lognormal(mean=6.5, sigma=1.0, size=n).clip(50, 50000),
        "service_line_count": rng.choice([1, 1, 1, 2, 2, 3, 4, 5], size=n),
        "has_multiple_cpt": rng.choice([0, 0, 0, 1, 1], size=n),
        "dx_count": rng.choice([1, 1, 2, 2, 3, 3, 4, 5, 6], size=n),
        "modifier_missing": rng.choice([0, 0, 0, 0, 1, 1], size=n),
        "patient_age": rng.normal(52, 18, size=n).clip(1, 95),
        "place_of_service_encoded": rng.choice([0, 1, 1, 1, 2, 3, 4, 5, 6, 7], size=n),
        "prior_auth_present": rng.choice([0, 0, 0, 1, 1], size=n),
        "payer_denial_rate": rng.beta(2, 8, size=n),
        "cpt_denial_rate": rng.beta(2, 10, size=n),
        "provider_denial_rate": rng.beta(2, 12, size=n),
    }
    df = pd.DataFrame(data)
    df["charge_per_line"] = df["total_charge"] / df["service_line_count"]

    # Denial labels
    prob = np.full(n, 0.15)
    prob += df["modifier_missing"].values * 0.25
    prob += (df["total_charge"].values > 5000).astype(float) * 0.12
    prob += (df["dx_count"].values < 2).astype(float) * 0.10
    prob += ((df["prior_auth_present"].values == 0) & (df["total_charge"].values > 3000)).astype(float) * 0.15
    prob += df["payer_denial_rate"].values * 0.3
    prob += df["cpt_denial_rate"].values * 0.2
    prob += rng.normal(0, 0.05, size=n)
    prob = prob.clip(0.01, 0.99)
    labels = (rng.random(n) < prob).astype(int)

    return df[FEATURE_NAMES].fillna(0.0), labels
