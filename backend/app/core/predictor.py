"""XGBoost model wrapper for denial prediction."""

import glob as glob_mod
import re
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

# Shadow model slot for champion-challenger comparison
_shadow_model = None
_shadow_model_loaded = False
_shadow_model_version: str = "none"


def _detect_version_from_files() -> str:
    """Scan model_v*.joblib files to detect latest version.

    Falls back to mtime-based version if no versioned files exist.
    """
    model_dir = Path(settings.MODEL_DIR)
    pattern = str(model_dir / "model_v*.joblib")
    versioned_files = glob_mod.glob(pattern)

    if versioned_files:
        # Extract version numbers and find the highest
        max_version = 0
        for f in versioned_files:
            match = re.search(r"model_v(\d+)\.joblib$", f)
            if match:
                v = int(match.group(1))
                if v > max_version:
                    max_version = v
        if max_version > 0:
            return f"v{max_version}"

    # Fallback: use mtime of demo_model.joblib
    demo_path = model_dir / "demo_model.joblib"
    if demo_path.exists():
        mtime = demo_path.stat().st_mtime
        from datetime import datetime
        return datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")

    return "fallback"


def set_model_version(version: str) -> None:
    """Set model version explicitly (called by retrain_model after saving)."""
    global _model_version
    _model_version = version
    logger.info("Model version set", version=version)


def load_model() -> None:
    """Load XGBoost model from disk."""
    global _model, _model_loaded, _model_version

    model_path = Path(settings.MODEL_DIR) / "demo_model.joblib"
    if model_path.exists():
        _model = joblib.load(model_path)
        _model_loaded = True
        _model_version = _detect_version_from_files()
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


def load_shadow_model(model_path: str, version: str) -> None:
    """Load a candidate model into the shadow slot for comparison scoring."""
    global _shadow_model, _shadow_model_loaded, _shadow_model_version

    path = Path(model_path)
    if path.exists():
        _shadow_model = joblib.load(path)
        _shadow_model_loaded = True
        _shadow_model_version = version
        logger.info("Shadow model loaded", path=str(path), version=version)
    else:
        logger.warning("Shadow model file not found", path=str(path))
        _shadow_model_loaded = False


def clear_shadow_model() -> None:
    """Clear the shadow model slot (e.g. after promotion)."""
    global _shadow_model, _shadow_model_loaded, _shadow_model_version
    _shadow_model = None
    _shadow_model_loaded = False
    _shadow_model_version = "none"
    logger.info("Shadow model cleared")


def is_shadow_loaded() -> bool:
    return _shadow_model_loaded


def get_shadow_version() -> str:
    return _shadow_model_version


def predict_shadow(X: pd.DataFrame) -> np.ndarray | None:
    """Score with the shadow model. Returns None if no shadow is loaded."""
    if _shadow_model is None:
        return None
    probas = _shadow_model.predict_proba(X)
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
