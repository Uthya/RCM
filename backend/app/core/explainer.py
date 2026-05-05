"""SHAP TreeExplainer for denial prediction model."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from app.config import settings
from app.core.feature_engineer import FEATURE_NAMES, FEATURE_DISPLAY_NAMES
from app.schemas.prediction import RiskFactor

logger = structlog.get_logger()

_explainer = None


def init_explainer(model) -> None:
    """Initialize SHAP TreeExplainer with the loaded model."""
    global _explainer
    try:
        import shap
        _explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized")
    except Exception as e:
        logger.warning("Could not initialize SHAP explainer", error=str(e))
        _explainer = None


def explain(X: pd.DataFrame, top_n: int = 3) -> list[list[RiskFactor]]:
    """Get top N SHAP risk factors for each prediction.
    Returns list of list of RiskFactor (one list per row in X).
    """
    if _explainer is None:
        return _fallback_explain(X, top_n)

    try:
        shap_values = _explainer.shap_values(X)

        # For binary classification, shap_values might be a list of 2 arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 (denied)

        results = []
        for i in range(len(X)):
            row_shap = shap_values[i]
            row_features = X.iloc[i]

            # Get top N by absolute SHAP value
            top_indices = np.argsort(np.abs(row_shap))[::-1][:top_n]

            factors = []
            for idx in top_indices:
                feat_name = FEATURE_NAMES[idx]
                factors.append(RiskFactor(
                    feature=feat_name,
                    display_name=FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
                    impact=round(float(row_shap[idx]), 4),
                    value=str(round(float(row_features.iloc[idx]), 4)),
                ))
            results.append(factors)

        return results

    except Exception as e:
        logger.warning("SHAP explanation failed, using fallback", error=str(e))
        return _fallback_explain(X, top_n)


def _fallback_explain(X: pd.DataFrame, top_n: int = 3) -> list[list[RiskFactor]]:
    """Rule-based fallback explanations when SHAP is unavailable."""
    results = []
    for i in range(len(X)):
        row = X.iloc[i]
        factors = []

        # Generate explanations based on feature values
        feature_impacts = []
        if row.get("modifier_missing", 0) == 1:
            feature_impacts.append(("modifier_missing", 0.25))
        if row.get("total_charge", 0) > 3000:
            feature_impacts.append(("total_charge", 0.18))
        if row.get("payer_denial_rate", 0) > 0.15:
            feature_impacts.append(("payer_denial_rate", 0.15))
        if row.get("dx_count", 0) < 2:
            feature_impacts.append(("dx_count", 0.12))
        if row.get("prior_auth_present", 0) == 0:
            feature_impacts.append(("prior_auth_present", 0.10))
        if row.get("cpt_denial_rate", 0) > 0.1:
            feature_impacts.append(("cpt_denial_rate", 0.08))
        if row.get("patient_age", 0) > 65:
            feature_impacts.append(("patient_age", 0.06))

        # Sort by impact and take top N
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat_name, impact in feature_impacts[:top_n]:
            factors.append(RiskFactor(
                feature=feat_name,
                display_name=FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
                impact=round(impact, 4),
                value=str(round(float(row.get(feat_name, 0)), 4)),
            ))

        # Pad with generic factors if needed
        while len(factors) < top_n:
            factors.append(RiskFactor(
                feature="base_rate",
                display_name="Base Denial Rate",
                impact=0.05,
                value="0.3",
            ))

        results.append(factors)

    return results
