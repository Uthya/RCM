"""Generate synthetic claims data and train a demo XGBoost denial prediction model."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier

# Feature names must match feature_engineer.py
FEATURE_NAMES = [
    "total_charge",
    "charge_per_line",
    "service_line_count",
    "has_multiple_cpt",
    "dx_count",
    "modifier_missing",
    "patient_age",
    "place_of_service_encoded",
    "prior_auth_present",
    "payer_denial_rate",
    "cpt_denial_rate",
    "provider_denial_rate",
]


def generate_synthetic_data(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic claims with realistic distributions."""
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
        "payer_denial_rate": rng.beta(2, 8, size=n),  # mean ~0.2
        "cpt_denial_rate": rng.beta(2, 10, size=n),   # mean ~0.17
        "provider_denial_rate": rng.beta(2, 12, size=n),  # mean ~0.14
    }

    df = pd.DataFrame(data)
    df["charge_per_line"] = df["total_charge"] / df["service_line_count"]

    # Generate denial labels based on realistic rules
    denial_prob = np.zeros(n)

    # Base denial rate ~15%
    denial_prob += 0.15

    # Missing modifier increases denial probability
    denial_prob += df["modifier_missing"] * 0.25

    # High charges increase denial
    denial_prob += (df["total_charge"] > 5000).astype(float) * 0.12
    denial_prob += (df["total_charge"] > 15000).astype(float) * 0.10

    # Fewer diagnoses (less support) increases denial
    denial_prob += (df["dx_count"] < 2).astype(float) * 0.10

    # No prior auth for expensive claims
    denial_prob += ((df["prior_auth_present"] == 0) & (df["total_charge"] > 3000)).astype(float) * 0.15

    # Payer historical denial rate influence
    denial_prob += df["payer_denial_rate"] * 0.3

    # CPT denial rate influence
    denial_prob += df["cpt_denial_rate"] * 0.2

    # Provider denial rate influence
    denial_prob += df["provider_denial_rate"] * 0.15

    # Emergency room visits less likely to be denied
    denial_prob -= (df["place_of_service_encoded"] == 4).astype(float) * 0.08

    # Age factor (very young or very old slightly higher)
    denial_prob += ((df["patient_age"] < 5) | (df["patient_age"] > 80)).astype(float) * 0.05

    # Add noise
    denial_prob += rng.normal(0, 0.05, size=n)

    # Clip and convert to binary
    denial_prob = denial_prob.clip(0.01, 0.99)
    df["denied"] = (rng.random(n) < denial_prob).astype(int)

    return df


def train_model():
    """Train XGBoost model on synthetic data and save artifacts."""
    print("Generating synthetic training data (10,000 claims)...")
    df = generate_synthetic_data(10000)

    X = df[FEATURE_NAMES]
    y = df["denied"]

    print(f"Denial rate in training data: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n--- Evaluation Metrics ---")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Paid", "Denied"]))

    # Save model
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "demo_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Feature importance
    print("\n--- Feature Importance ---")
    importance = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
        print(f"  {name:30s} {imp:.4f}")


if __name__ == "__main__":
    train_model()
