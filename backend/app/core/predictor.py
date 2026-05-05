"""XGBoost model wrapper for denial prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import structlog

from app.config import settings

logger = structlog.get_logger()

_model = None
_model_loaded = False
_model_version: str = "unknown"


def load_model() -> None:
    """Load XGBoost model from disk."""
    global _model, _model_loaded, _model_version

    model_path = Path(settings.MODEL_DIR) / "demo_model.joblib"
    if model_path.exists():
        _model = joblib.load(model_path)
        _model_loaded = True
        # Track model version from file modification time
        mtime = model_path.stat().st_mtime
        from datetime import datetime
        _model_version = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
        logger.info("XGBoost model loaded", path=str(model_path), version=_model_version)
        # Initialize SHAP explainer
        try:
            from app.core.explainer import init_explainer
            init_explainer(_model)
        except Exception as e:
            logger.warning("Could not init SHAP explainer", error=str(e))
    else:
        logger.warning("No model file found, predictions will use fallback", path=str(model_path))
        _model_loaded = False
        _model_version = "fallback"


def is_model_loaded() -> bool:
    return _model_loaded


def get_model_version() -> str:
    return _model_version


def predict_proba(X: pd.DataFrame) -> np.ndarray:
    """Predict denial probability for feature matrix.
    Returns array of denial probabilities.
    """
    if _model is None:
        # Fallback: return random-ish scores based on features for demo
        logger.warning("Using fallback prediction (no model loaded)")
        return _fallback_predict(X)

    probas = _model.predict_proba(X)
    # Return probability of class 1 (denied)
    return probas[:, 1]


def _fallback_predict(X: pd.DataFrame) -> np.ndarray:
    """Simple rule-based fallback when no trained model is available."""
    scores = np.zeros(len(X))
    for i in range(len(X)):
        row = X.iloc[i]
        score = 0.3  # base

        if row.get("modifier_missing", 0) == 1:
            score += 0.2
        if row.get("total_charge", 0) > 5000:
            score += 0.15
        if row.get("dx_count", 0) < 2:
            score += 0.1
        if row.get("payer_denial_rate", 0) > 0.2:
            score += 0.15
        if row.get("prior_auth_present", 0) == 0 and row.get("total_charge", 0) > 1000:
            score += 0.1

        scores[i] = min(score, 0.99)

    return scores
