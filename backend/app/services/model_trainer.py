"""
Retrain XGBoost model using pre-joined ml_training_data from PostgreSQL.

The ml_training_data table contains prediction features joined with
835 outcomes — no feature recomputation needed. Model versions are tracked
in the model_registry table.
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

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.feature_engineer import FEATURE_NAMES, FEATURE_VERSION
from app.repositories import training_repo, model_repo

logger = structlog.get_logger()

MIN_REAL_SAMPLES = 30
SYNTHETIC_FILL_TARGET = 2000
AUTO_RETRAIN_THRESHOLD = 5000
AUTO_RETRAIN_INTERVAL_DAYS = 7
DEFAULT_TRAINING_WINDOW_DAYS = 180
ARCHIVE_AFTER_DAYS = 365


async def retrain_model(session: AsyncSession, training_window_days: int = DEFAULT_TRAINING_WINDOW_DAYS) -> dict:
    """
    Read pre-joined training data from ml_training_data, train XGBoost,
    save versioned model. Returns training summary dict.
    """
    start = time.time()

    # 1. Pull training records (time-windowed + version-filtered)
    cutoff = datetime.utcnow() - timedelta(days=training_window_days)
    real_docs = await training_repo.get_training_data(
        session, created_after=cutoff, feature_version=FEATURE_VERSION,
        first_attempt_only=True,
    )
    real_count = len(real_docs)
    used_window = True

    # Fallback: without version filter
    if real_count < MIN_REAL_SAMPLES:
        logger.warning("Retrain: version-filtered windowed data insufficient",
                       windowed_count=real_count, feature_version=FEATURE_VERSION)
        real_docs = await training_repo.get_training_data(
            session, created_after=cutoff, first_attempt_only=True,
        )
        real_count = len(real_docs)

    # Fallback: all records
    if real_count < MIN_REAL_SAMPLES:
        logger.warning("Retrain: windowed data insufficient, falling back to all records",
                       windowed_count=real_count)
        real_docs = await training_repo.get_training_data(
            session, first_attempt_only=True,
        )
        real_count = len(real_docs)
        used_window = False

    logger.info("Retrain: found training records", count=real_count,
                window_days=training_window_days if used_window else None, used_window=used_window)

    if real_count < MIN_REAL_SAMPLES:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {MIN_REAL_SAMPLES} training records (have {real_count}). "
                       f"Upload more 835 files to build training data.",
            "training_records": real_count,
        }

    # 2. Use pre-joined features + labels directly
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

    # Collect top denial codes
    denial_code_counts: dict[str, int] = {}
    for doc in real_docs:
        code = doc.get("denial_code")
        if code and doc.get("label") == 1:
            denial_code_counts[code] = denial_code_counts.get(code, 0) + 1
    top_denial_codes = sorted(denial_code_counts.items(), key=lambda x: -x[1])[:10]

    # 3. Augment with synthetic if needed
    synthetic_count = 0
    if real_count < SYNTHETIC_FILL_TARGET:
        synthetic_count = SYNTHETIC_FILL_TARGET - real_count
        df_synth, y_synth = _generate_synthetic(synthetic_count, seed=int(time.time()) % 10000)
    else:
        df_synth, y_synth = None, None

    # 4. Train/test split — temporal for real data when possible
    temporal_split = False
    if real_count >= 50:
        # Temporal split: oldest 80% train, newest 20% test (data is ordered by created_at)
        split_idx = int(real_count * 0.8)
        df_real_train = df_real.iloc[:split_idx]
        y_real_train = y_real[:split_idx]
        df_real_test = df_real.iloc[split_idx:]
        y_real_test = y_real[split_idx:]

        # Synthetic only augments training set
        if df_synth is not None:
            X_train = pd.concat([df_real_train, df_synth], ignore_index=True)
            y_tr = np.concatenate([y_real_train, y_synth])
        else:
            X_train = df_real_train
            y_tr = y_real_train

        # Test set is ONLY real data (newest 20%)
        X_test = df_real_test
        y_te = y_real_test
        temporal_split = True
        real_test_flags = np.ones(len(y_te), dtype=bool)  # all test data is real
    else:
        # Fallback: random split with real+synthetic
        if df_synth is not None:
            df_combined = pd.concat([df_real, df_synth], ignore_index=True)
            y_combined = np.concatenate([y_real, y_synth])
        else:
            df_combined = df_real
            y_combined = y_real

        # Track real vs synthetic through the split
        is_real = np.array([True] * real_count + [False] * synthetic_count)

        X_train, X_test, y_tr, y_te, real_train_flags, real_test_flags = train_test_split(
            df_combined, y_combined, is_real,
            test_size=0.2, random_state=42, stratify=y_combined,
        )

    # 5. Train XGBoost
    pos_weight = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_tr, eval_set=[(X_test, y_te)], verbose=False)

    # 6. Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_te, y_pred_proba))
    precision = float(precision_score(y_te, y_pred, zero_division=0))
    recall = float(recall_score(y_te, y_pred, zero_division=0))

    # Real-only metrics (correctly tracked through split)
    real_test_count = int(real_test_flags.sum())
    if real_test_count > 10:
        auc_real = float(roc_auc_score(y_te[real_test_flags], y_pred_proba[real_test_flags]))
        precision_real = float(precision_score(y_te[real_test_flags], y_pred[real_test_flags], zero_division=0))
        recall_real = float(recall_score(y_te[real_test_flags], y_pred[real_test_flags], zero_division=0))
    else:
        auc_real = None
        precision_real = None
        recall_real = None

    # 7. Save versioned model
    model_dir = Path(settings.MODEL_DIR)
    model_dir.mkdir(exist_ok=True)

    latest_ver = await model_repo.get_latest_version(session)
    version_num = (latest_ver + 1) if latest_ver else 1
    version_str = f"v{version_num}"
    versioned_path = model_dir / f"model_{version_str}.joblib"
    joblib.dump(model, versioned_path)

    # 8. Save training record + model registry
    elapsed = round(time.time() - start, 2)

    metrics = {
        "auc_roc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "auc_real_only": round(auc_real, 4) if auc_real else None,
        "precision_real_only": round(precision_real, 4) if precision_real else None,
        "recall_real_only": round(recall_real, 4) if recall_real else None,
        "temporal_split": temporal_split,
        "real_test_count": real_test_count,
    }
    feature_importance = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, model.feature_importances_)
    }

    await training_repo.insert_history(session, {
        "trained_at": datetime.utcnow(),
        "model_version": version_str,
        "real_samples": real_count,
        "synthetic_samples": synthetic_count,
        "total_samples": len(X_train),
        "denied_count": denied_count,
        "paid_count": paid_count,
        "denial_rate": round(denied_count / real_count, 4),
        "training_window_days": training_window_days if used_window else None,
        "used_full_dataset": not used_window,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "top_denial_codes": [{"code": c, "count": n} for c, n in top_denial_codes],
        "elapsed_seconds": elapsed,
    })

    from app.core.feature_engineer import FEATURE_COUNT, FEATURE_HASH

    await model_repo.upsert_model(session, {
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
        "feature_version": FEATURE_VERSION,
        "feature_count": FEATURE_COUNT,
        "feature_hash": FEATURE_HASH,
        "top_denial_codes": [{"code": c, "count": n} for c, n in top_denial_codes],
    })

    await session.commit()

    # 9. Archive old training data
    archive_cutoff = datetime.utcnow() - timedelta(days=ARCHIVE_AFTER_DAYS)
    archived_count = await training_repo.archive_old(session, archive_cutoff)
    if archived_count:
        await session.commit()
        logger.info("Retrain: archived old training data", archived=archived_count)

    # 10. Load as shadow model
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
        "archived_records": archived_count,
        "elapsed_seconds": elapsed,
    }


async def get_training_status(session: AsyncSession) -> dict:
    from app.repositories import claim_repo, remittance_repo, outcome_repo
    total_claims = await claim_repo.count_claims(session)
    total_remittances = await remittance_repo.count_remittances(session)
    matched = await outcome_repo.count_outcomes(session, status_filter=["paid", "denied"])

    training_total = await training_repo.count_records(session)
    training_denied = await training_repo.count_records(session, label=1)
    training_paid = training_total - training_denied
    archived_records = await training_repo.count_archived(session)

    active_model = await model_repo.get_active(session)

    gap = matched - training_total

    last = await training_repo.get_latest_history(session)

    return {
        "total_claims": total_claims,
        "total_remittances": total_remittances,
        "matched_claims": matched,
        "training_data": {
            "total": training_total,
            "denied": training_denied,
            "paid": training_paid,
            "denial_rate": round(training_denied / max(training_total, 1), 4),
            "archived_records": archived_records,
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
            "trained_at": active_model["trained_at"].isoformat() if active_model.get("trained_at") else None,
            "metrics": active_model.get("metrics"),
        } if active_model else None,
        "ready_to_train": training_total >= MIN_REAL_SAMPLES,
        "min_required": MIN_REAL_SAMPLES,
        "last_training": {
            "trained_at": last["trained_at"].isoformat() if last and last.get("trained_at") else None,
            "model_version": last.get("model_version") if last else None,
            "real_samples": last.get("real_samples") if last else None,
            "metrics": last.get("metrics") if last else None,
        } if last else None,
    }


async def validate_training_data(session: AsyncSession) -> dict:
    issues = []

    total = await training_repo.count_records(session)
    denied = await training_repo.count_records(session, label=1)
    paid = total - denied
    denial_rate = denied / max(total, 1)

    if denial_rate < 0.05 or denial_rate > 0.95:
        issues.append(f"Extreme class imbalance: {denial_rate:.1%} denial rate")

    empty_feats = await training_repo.count_empty_features(session)
    if empty_feats > total * 0.1:
        issues.append(f"{empty_feats} records have empty features (>{10}% of data)")

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


AUC_DROP_WARN_THRESHOLD = 0.02


async def promote_model(session: AsyncSession, version: int, *, force: bool = False) -> dict:
    candidate = await model_repo.find_by_version(session, version)
    if not candidate:
        return {"status": "error", "message": f"Model version {version} not found"}

    if candidate.get("is_active"):
        return {"status": "error", "message": f"Model v{version} is already active"}

    active = await model_repo.get_active(session)

    if active and not force:
        active_auc = (active.get("metrics") or {}).get("auc_roc", 0)
        candidate_auc = (candidate.get("metrics") or {}).get("auc_roc", 0)
        if active_auc - candidate_auc > AUC_DROP_WARN_THRESHOLD:
            return {
                "status": "warning",
                "message": (
                    f"Candidate v{version} AUC ({candidate_auc:.4f}) is lower than "
                    f"active v{active['version']} AUC ({active_auc:.4f}) by "
                    f"{active_auc - candidate_auc:.4f}. Pass force=true to promote anyway."
                ),
                "active_metrics": active.get("metrics"),
                "candidate_metrics": candidate.get("metrics"),
            }

    await model_repo.deactivate_all(session)
    await model_repo.activate(session, version)
    await session.commit()

    model_path = candidate.get("model_path")
    if model_path:
        demo_path = Path(settings.MODEL_DIR) / "demo_model.joblib"
        shutil.copy2(model_path, str(demo_path))

    from app.core.predictor import load_model, set_model_version, clear_shadow_model
    load_model()
    set_model_version(f"v{version}")
    clear_shadow_model()

    from app.services.decision_engine import load_config as load_decision_config
    await load_decision_config(session)

    logger.info("Model promoted", version=version)

    return {
        "status": "success",
        "message": f"Model v{version} promoted to active",
        "promoted_version": f"v{version}",
        "previous_active": f"v{active['version']}" if active else None,
        "metrics": candidate.get("metrics"),
    }


async def rollback_model(session: AsyncSession) -> dict:
    active = await model_repo.get_active(session)
    if not active:
        return {"status": "error", "message": "No active model to rollback from"}

    previous = await model_repo.get_previous_retired(session)
    if not previous:
        return {"status": "error", "message": "No previous model version to rollback to"}

    await model_repo.set_status(session, active["version"], "rolled_back")
    await model_repo.activate(session, previous["version"])
    await session.commit()

    model_path = previous.get("model_path")
    if model_path:
        demo_path = Path(settings.MODEL_DIR) / "demo_model.joblib"
        shutil.copy2(model_path, str(demo_path))

    from app.core.predictor import load_model, set_model_version, clear_shadow_model
    load_model()
    set_model_version(f"v{previous['version']}")
    clear_shadow_model()

    from app.services.decision_engine import load_config as load_decision_config
    await load_decision_config(session)

    logger.info("Model rolled back", from_version=active["version"], to_version=previous["version"])

    return {
        "status": "success",
        "message": f"Rolled back from v{active['version']} to v{previous['version']}",
        "rolled_back_from": f"v{active['version']}",
        "restored_version": f"v{previous['version']}",
        "metrics": previous.get("metrics"),
    }


def _generate_synthetic(n: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    data = {
        # Existing 14 features
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
        "invalid_npi": rng.choice([0, 0, 0, 0, 0, 1], size=n),
        "duplicate_risk": rng.choice([0.0, 0.0, 0.0, 0.0, 0.33], size=n),
        # Aggregate confidence
        "payer_denial_rate_n": rng.choice([0, 0, 5, 10, 20, 50, 100, 200], size=n),
        "cpt_denial_rate_n": rng.choice([0, 0, 5, 10, 20, 50, 100], size=n),
        "provider_denial_rate_n": rng.choice([0, 0, 5, 10, 20, 50], size=n),
        # New claim-level features
        "modifier_count": rng.choice([0, 0, 1, 1, 1, 2, 2, 3], size=n),
        "dx_specificity": rng.normal(4.5, 1.0, size=n).clip(3, 7),
        "cpt_category": rng.choice([0, 1, 1, 1, 2, 2, 3, 4, 5], size=n),
        "patient_gender_encoded": rng.choice([0, 1, 1, 2, 2], size=n),
        "has_rendering_provider": rng.choice([0, 0, 1, 1, 1], size=n),
        "taxonomy_category": rng.choice([0, 0, 1, 1, 2, 3, 4, 5], size=n),
        "frequency_code_encoded": rng.choice([0, 1, 1, 1, 1, 7], size=n),
        "payer_sequence_encoded": rng.choice([0, 1, 1, 1, 2], size=n),
        "filing_lag_days": rng.lognormal(mean=3.0, sigma=0.8, size=n).clip(0, 365),
        "charge_dx_ratio": rng.lognormal(mean=5.5, sigma=1.2, size=n).clip(10, 50000),
    }
    df = pd.DataFrame(data)
    df["charge_per_line"] = df["total_charge"] / df["service_line_count"]

    # Label generation — reduced coefficients + increased noise
    prob = np.full(n, 0.15)
    prob += df["modifier_missing"].values * 0.15
    prob += (df["total_charge"].values > 5000).astype(float) * 0.08
    prob += (df["dx_count"].values < 2).astype(float) * 0.06
    prob += ((df["prior_auth_present"].values == 0) & (df["total_charge"].values > 3000)).astype(float) * 0.10
    prob += df["payer_denial_rate"].values * 0.15
    prob += df["cpt_denial_rate"].values * 0.10
    prob += data["invalid_npi"] * 0.10
    prob += (np.array(data["duplicate_risk"]) > 0).astype(float) * 0.12
    # New feature contributions
    prob += (df["dx_specificity"].values < 4).astype(float) * 0.05
    prob += (df["modifier_count"].values == 0).astype(float) * 0.05
    prob += (df["filing_lag_days"].values > 90).astype(float) * 0.08
    prob += rng.normal(0, 0.10, size=n)  # increased noise (was 0.05)
    prob = prob.clip(0.05, 0.95)  # tighter clip (was 0.01, 0.99)
    labels = (rng.random(n) < prob).astype(int)

    return df[FEATURE_NAMES].fillna(0.0), labels
