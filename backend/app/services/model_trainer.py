"""
Retrain XGBoost model using pre-joined ml_training_data from MongoDB.

The ml_training_data collection contains prediction features joined with
835 outcomes — no feature recomputation needed. Model versions are tracked
in the model_registry collection.
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier

from app.config import settings
from app.core.feature_engineer import FEATURE_NAMES

logger = structlog.get_logger()

MIN_REAL_SAMPLES = 30       # minimum real training records to attempt training
SYNTHETIC_FILL_TARGET = 2000  # if real < this, pad with synthetic
AUTO_RETRAIN_THRESHOLD = 5000  # training records to trigger auto-retrain
AUTO_RETRAIN_INTERVAL_DAYS = 7  # minimum days between auto-retrains
DEFAULT_TRAINING_WINDOW_DAYS = 180  # default time window for training data


async def _get_next_model_version(db) -> int:
    """Read latest version from model_registry, return next int (1 if first)."""
    latest = await db.model_registry.find_one(
        sort=[("version", -1)],
    )
    if latest:
        return latest["version"] + 1
    return 1


async def retrain_model(db, training_window_days: int = DEFAULT_TRAINING_WINDOW_DAYS) -> dict:
    """
    Read pre-joined training data from ml_training_data, train XGBoost,
    save versioned model. Returns training summary dict.

    Args:
        training_window_days: Only use training records created within this many
            days. Falls back to all data if the windowed set is too small.
    """
    start = time.time()

    # ── 1. Pull training records from ml_training_data (time-windowed) ──
    cutoff = datetime.utcnow() - timedelta(days=training_window_days)
    windowed_query = {"created_at": {"$gte": cutoff}}
    cursor = db.ml_training_data.find(windowed_query)
    real_docs = await cursor.to_list(length=100_000)
    real_count = len(real_docs)
    used_window = True

    # Fallback: if windowed data is too small, use all records
    if real_count < MIN_REAL_SAMPLES:
        logger.warning(
            "Retrain: windowed data insufficient, falling back to all records",
            windowed_count=real_count,
            window_days=training_window_days,
        )
        cursor = db.ml_training_data.find({})
        real_docs = await cursor.to_list(length=100_000)
        real_count = len(real_docs)
        used_window = False

    logger.info("Retrain: found training records",
                count=real_count,
                window_days=training_window_days if used_window else None,
                used_window=used_window)

    if real_count < MIN_REAL_SAMPLES:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {MIN_REAL_SAMPLES} training records (have {real_count}). "
                       f"Upload more 835 files to build training data.",
            "training_records": real_count,
        }

    # ── 2. Use pre-joined features + labels directly (no recomputation) ──
    real_features: list[dict] = []
    real_labels: list[int] = []
    skipped = 0

    for doc in real_docs:
        features = doc.get("features")
        if not features:
            skipped += 1
            continue
        real_features.append(features)
        real_labels.append(doc["label"])

    if skipped:
        logger.warning("Retrain: skipped records with empty features", count=skipped)

    real_count = len(real_features)
    if real_count < MIN_REAL_SAMPLES:
        return {
            "status": "insufficient_data",
            "message": f"Only {real_count} records have features (need {MIN_REAL_SAMPLES}).",
            "training_records": real_count,
        }

    df_real = pd.DataFrame(real_features, columns=FEATURE_NAMES).fillna(0.0)
    y_real = np.array(real_labels)

    denied_count = int(y_real.sum())
    paid_count = int(len(y_real) - denied_count)

    # Collect top denial codes for training record
    denial_code_counts: dict[str, int] = {}
    for doc in real_docs:
        code = doc.get("denial_code")
        if code and doc.get("label") == 1:
            denial_code_counts[code] = denial_code_counts.get(code, 0) + 1
    top_denial_codes = sorted(denial_code_counts.items(), key=lambda x: -x[1])[:10]

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

    # ── 7. Save versioned model ──
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(exist_ok=True)

    version_num = await _get_next_model_version(db)
    version_str = f"v{version_num}"
    versioned_path = model_dir / f"model_{version_str}.joblib"
    joblib.dump(model, versioned_path)
    logger.info("Retrain: saved versioned model", path=str(versioned_path), version=version_str)

    # NOTE: No longer copy to demo_model.joblib here — that happens on promote_model()

    # ── 8. Save training record + model registry ──
    elapsed = round(time.time() - start, 2)

    metrics = {
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "auc_real_only": round(auc_real, 4) if auc_real else None,
    }
    feature_importance = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, model.feature_importances_)
    }

    training_record = {
        "trained_at": datetime.utcnow(),
        "model_version": version_str,
        "real_samples": real_count,
        "synthetic_samples": synthetic_count,
        "total_samples": len(df_train),
        "denied_count": denied_count,
        "paid_count": paid_count,
        "denial_rate": round(denied_count / real_count, 4),
        "training_window_days": training_window_days if used_window else None,
        "used_full_dataset": not used_window,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "top_denial_codes": [{"code": c, "count": n} for c, n in top_denial_codes],
        "elapsed_seconds": elapsed,
    }
    await db.training_history.insert_one(training_record)

    # Register new model as candidate (NOT active) — requires explicit promotion
    await db.model_registry.update_one(
        {"version": version_num},
        {"$set": {
            "version": version_num,
            "version_str": version_str,
            "trained_at": datetime.utcnow(),
            "is_active": False,
            "status": "candidate",
            "model_path": str(versioned_path),
            "real_samples": real_count,
            "synthetic_samples": synthetic_count,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "top_denial_codes": [{"code": c, "count": n} for c, n in top_denial_codes],
        }},
        upsert=True,
    )

    logger.info("Retrain: model saved as candidate (not promoted)",
                version=version_str)

    # ── 9. Load as shadow model for comparison scoring ──
    from app.core.predictor import load_shadow_model
    load_shadow_model(str(versioned_path), version_str)

    window_note = f" (last {training_window_days} days)" if used_window else " (all data — window fallback)"
    return {
        "status": "candidate",
        "model_version": version_str,
        "message": (
            f"Model {version_str} trained on {real_count} real + "
            f"{synthetic_count} synthetic records{window_note}. "
            f"Loaded as shadow model — call POST /model/promote/{version_num} to activate."
        ),
        "real_samples": real_count,
        "synthetic_samples": synthetic_count,
        "denied_count": denied_count,
        "paid_count": paid_count,
        "denial_rate": round(denied_count / real_count, 4),
        "training_window_days": training_window_days if used_window else None,
        "used_full_dataset": not used_window,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "top_denial_codes": [{"code": c, "count": n} for c, n in top_denial_codes],
        "elapsed_seconds": elapsed,
    }


async def get_training_status(db) -> dict:
    """Get current training data status and last training info."""
    total_claims = await db.claims.count_documents({})
    total_remittances = await db.remittances.count_documents({})
    matched = await db.claims.count_documents(
        {"actual_outcome": {"$in": ["paid", "denied"]}}
    )

    # ml_training_data stats
    training_total = await db.ml_training_data.count_documents({})
    training_denied = await db.ml_training_data.count_documents({"label": 1})
    training_paid = training_total - training_denied

    # Active model version
    active_model = await db.model_registry.find_one({"is_active": True})

    # Gap analysis: matched claims vs training records
    gap = matched - training_total

    # Last training
    last = await db.training_history.find_one(sort=[("trained_at", -1)])

    return {
        "total_claims": total_claims,
        "total_remittances": total_remittances,
        "matched_claims": matched,
        "training_data": {
            "total": training_total,
            "denied": training_denied,
            "paid": training_paid,
            "denial_rate": round(training_denied / max(training_total, 1), 4),
        },
        "gap_analysis": {
            "matched_claims": matched,
            "training_records": training_total,
            "gap": gap,
            "message": f"{gap} matched claims not yet in training data" if gap > 0
                       else "All matched claims have training records",
        },
        "active_model": {
            "version": active_model["version_str"],
            "trained_at": active_model["trained_at"].isoformat(),
            "metrics": active_model.get("metrics"),
        } if active_model else None,
        "ready_to_train": training_total >= MIN_REAL_SAMPLES,
        "min_required": MIN_REAL_SAMPLES,
        "last_training": {
            "trained_at": last["trained_at"].isoformat() if last else None,
            "model_version": last.get("model_version") if last else None,
            "real_samples": last.get("real_samples") if last else None,
            "metrics": last.get("metrics") if last else None,
        } if last else None,
    }


async def validate_training_data(db) -> dict:
    """Validate data quality BEFORE allowing retrain.

    Reads from ml_training_data instead of claims.
    Prevents noisy labels from degrading the model. All gates must pass.
    """
    issues = []

    # 1. Check class balance
    total = await db.ml_training_data.count_documents({})
    denied = await db.ml_training_data.count_documents({"label": 1})
    paid = total - denied
    denial_rate = denied / max(total, 1)

    if denial_rate < 0.05 or denial_rate > 0.95:
        issues.append(f"Extreme class imbalance: {denial_rate:.1%} denial rate")

    # 2. Check for empty features
    empty_feats = await db.ml_training_data.count_documents({
        "$or": [
            {"features": {"$exists": False}},
            {"features": {}},
        ],
    })
    if empty_feats > total * 0.1:
        issues.append(f"{empty_feats} records have empty features (>{10}% of data)")

    # 3. Minimum sample size
    if total < MIN_REAL_SAMPLES:
        issues.append(f"Only {total} training records (need {MIN_REAL_SAMPLES})")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "stats": {
            "total": total,
            "denied": denied,
            "paid": paid,
            "denial_rate": round(denial_rate, 4),
        },
    }


AUC_DROP_WARN_THRESHOLD = 0.02  # warn if new model AUC drops more than this


async def promote_model(db, version: int, *, force: bool = False) -> dict:
    """Promote a candidate model to active after metric comparison.

    Returns a warning (not error) if AUC drops > threshold unless force=True.
    """
    candidate = await db.model_registry.find_one({"version": version})
    if not candidate:
        return {"status": "error", "message": f"Model version {version} not found"}

    if candidate.get("is_active"):
        return {"status": "error", "message": f"Model v{version} is already active"}

    active = await db.model_registry.find_one({"is_active": True})

    # Metric gate: warn on AUC regression
    if active and not force:
        active_auc = (active.get("metrics") or {}).get("auc_roc", 0)
        candidate_auc = (candidate.get("metrics") or {}).get("auc_roc", 0)
        if active_auc - candidate_auc > AUC_DROP_WARN_THRESHOLD:
            return {
                "status": "warning",
                "message": (
                    f"Candidate v{version} AUC ({candidate_auc:.4f}) is lower than "
                    f"active v{active['version']} AUC ({active_auc:.4f}) by "
                    f"{active_auc - candidate_auc:.4f}. "
                    f"Pass force=true to promote anyway."
                ),
                "active_metrics": active.get("metrics"),
                "candidate_metrics": candidate.get("metrics"),
            }

    # Deactivate current active model
    await db.model_registry.update_many(
        {"is_active": True},
        {"$set": {"is_active": False, "status": "retired"}},
    )

    # Activate the candidate
    await db.model_registry.update_one(
        {"version": version},
        {"$set": {"is_active": True, "status": "active", "promoted_at": datetime.utcnow()}},
    )

    # Copy model file to demo_model.joblib and reload in memory
    model_path = candidate.get("model_path")
    if model_path:
        demo_path = Path(settings.MODEL_DIR) / "demo_model.joblib"
        shutil.copy2(model_path, str(demo_path))

    from app.core.predictor import load_model, set_model_version, clear_shadow_model
    load_model()
    set_model_version(f"v{version}")
    clear_shadow_model()

    from app.services.decision_engine import load_config as load_decision_config
    await load_decision_config(db)

    logger.info("Model promoted", version=version)

    return {
        "status": "success",
        "message": f"Model v{version} promoted to active",
        "promoted_version": f"v{version}",
        "previous_active": f"v{active['version']}" if active else None,
        "metrics": candidate.get("metrics"),
    }


async def rollback_model(db) -> dict:
    """Rollback to the previous active model version."""
    # Find the current active model
    active = await db.model_registry.find_one({"is_active": True})
    if not active:
        return {"status": "error", "message": "No active model to rollback from"}

    # Find the most recent retired model (previous active)
    previous = await db.model_registry.find_one(
        {"status": "retired"},
        sort=[("version", -1)],
    )
    if not previous:
        return {"status": "error", "message": "No previous model version to rollback to"}

    # Deactivate current
    await db.model_registry.update_one(
        {"version": active["version"]},
        {"$set": {"is_active": False, "status": "rolled_back"}},
    )

    # Reactivate previous
    await db.model_registry.update_one(
        {"version": previous["version"]},
        {"$set": {"is_active": True, "status": "active", "promoted_at": datetime.utcnow()}},
    )

    # Copy previous model file and reload
    model_path = previous.get("model_path")
    if model_path:
        demo_path = Path(settings.MODEL_DIR) / "demo_model.joblib"
        shutil.copy2(model_path, str(demo_path))

    from app.core.predictor import load_model, set_model_version, clear_shadow_model
    load_model()
    set_model_version(f"v{previous['version']}")
    clear_shadow_model()

    from app.services.decision_engine import load_config as load_decision_config
    await load_decision_config(db)

    logger.info("Model rolled back",
                from_version=active["version"],
                to_version=previous["version"])

    return {
        "status": "success",
        "message": f"Rolled back from v{active['version']} to v{previous['version']}",
        "rolled_back_from": f"v{active['version']}",
        "restored_version": f"v{previous['version']}",
        "metrics": previous.get("metrics"),
    }


# ── Helpers ──

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
