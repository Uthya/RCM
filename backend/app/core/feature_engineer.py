"""Transform parsed claim JSON into numeric feature vectors for XGBoost.

Uses Redis for caching historical denial rates (TTL = 5 min).
Falls back to direct PostgreSQL queries if Redis is unavailable.
"""

from datetime import datetime

import hashlib
import numpy as np
import pandas as pd
import structlog

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.repositories import remittance_repo

logger = structlog.get_logger()

# ── Redis client (initialized lazily) ──

_redis = None
_redis_available = False
CACHE_TTL = 300  # 5 minutes


async def init_cache():
    """Initialize Redis connection. Called at startup."""
    global _redis, _redis_available
    try:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        await _redis.ping()
        _redis_available = True
        logger.info("Redis cache connected", url=settings.REDIS_URL)
    except Exception as e:
        logger.warning("Redis unavailable, using direct DB queries", error=str(e))
        _redis_available = False


async def _get_cached_rate(key: str) -> float | None:
    if _redis and _redis_available:
        try:
            val = await _redis.get(key)
            if val is not None:
                return float(val)
        except Exception:
            pass
    return None


async def _set_cached_rate(key: str, rate: float):
    if _redis and _redis_available:
        try:
            await _redis.set(key, str(rate), ex=CACHE_TTL)
        except Exception:
            pass


# Bump this whenever FEATURE_NAMES list changes (add, remove, rename, reorder)
FEATURE_VERSION = "v1"

# Ordered feature names matching the trained model
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

FEATURE_COUNT = len(FEATURE_NAMES)
FEATURE_HASH = hashlib.sha256(",".join(sorted(FEATURE_NAMES)).encode()).hexdigest()[:16]

# Feature display names for SHAP explanations
FEATURE_DISPLAY_NAMES = {
    "total_charge": "Total Charge Amount",
    "charge_per_line": "Charge Per Service Line",
    "service_line_count": "Number of Service Lines",
    "has_multiple_cpt": "Multiple CPT Codes",
    "dx_count": "Diagnosis Code Count",
    "modifier_missing": "Missing Modifier",
    "patient_age": "Patient Age",
    "place_of_service_encoded": "Place of Service",
    "prior_auth_present": "Prior Authorization Present",
    "payer_denial_rate": "Payer Historical Denial Rate",
    "cpt_denial_rate": "CPT Historical Denial Rate",
    "provider_denial_rate": "Provider Historical Denial Rate",
}


def compute_features_from_claim(claim_doc: dict) -> dict:
    """Compute feature vector from a claim document (no async DB calls)."""
    service_lines = claim_doc.get("service_lines", [])
    total_charge = claim_doc.get("total_charge", 0.0)
    line_count = max(len(service_lines), 1)

    modifier_missing = 0
    cpt_codes = set()
    for sl in service_lines:
        mods = sl.get("modifiers", [])
        if not mods or all(m == "" for m in mods):
            modifier_missing = 1
        cpt_codes.add(sl.get("cpt_code", ""))

    patient_age = _compute_age(claim_doc.get("patient_dob", ""))
    pos = claim_doc.get("place_of_service", "11")
    pos_encoded = _encode_pos(pos)
    prior_auth = 1 if claim_doc.get("prior_auth_number", "") else 0

    return {
        "total_charge": total_charge,
        "charge_per_line": total_charge / line_count,
        "service_line_count": line_count,
        "has_multiple_cpt": 1 if len(cpt_codes) > 1 else 0,
        "dx_count": len(claim_doc.get("diagnosis_codes", [])),
        "modifier_missing": modifier_missing,
        "patient_age": patient_age,
        "place_of_service_encoded": pos_encoded,
        "prior_auth_present": prior_auth,
        "payer_denial_rate": 0.0,
        "cpt_denial_rate": 0.0,
        "provider_denial_rate": 0.0,
    }


async def enrich_with_historical_rates(session: AsyncSession, features: dict, claim_doc: dict) -> dict:
    """Enrich features with historical denial rates. Uses Redis cache first."""
    payer_name = claim_doc.get("payer_name", "")
    payer_id = claim_doc.get("payer_id", "")
    if payer_id:
        cache_key = f"denial_rate:payer:{payer_name}"
        cached = await _get_cached_rate(cache_key)
        if cached is not None:
            features["payer_denial_rate"] = cached
        else:
            rate = await remittance_repo.get_denial_rate(session, "payer_name", payer_name)
            features["payer_denial_rate"] = rate
            await _set_cached_rate(cache_key, rate)

    service_lines = claim_doc.get("service_lines", [])
    if service_lines:
        primary_cpt = service_lines[0].get("cpt_code", "")
        if primary_cpt:
            cache_key = f"denial_rate:cpt:{primary_cpt}"
            cached = await _get_cached_rate(cache_key)
            if cached is not None:
                features["cpt_denial_rate"] = cached
            else:
                rate = await remittance_repo.get_cpt_denial_rate(session, primary_cpt)
                features["cpt_denial_rate"] = rate
                await _set_cached_rate(cache_key, rate)

    npi = claim_doc.get("billing_provider_npi", "")
    if npi:
        cache_key = f"denial_rate:provider:{npi}"
        cached = await _get_cached_rate(cache_key)
        if cached is not None:
            features["provider_denial_rate"] = cached
        else:
            rate = await remittance_repo.get_denial_rate(session, "payee_npi", npi)
            features["provider_denial_rate"] = rate
            await _set_cached_rate(cache_key, rate)

    return features


def features_to_dataframe(feature_dicts: list[dict]) -> pd.DataFrame:
    """Convert list of feature dicts to a DataFrame with ordered columns."""
    return pd.DataFrame(feature_dicts, columns=FEATURE_NAMES).fillna(0.0)


def _compute_age(dob_str: str) -> float:
    if not dob_str or len(dob_str) < 8:
        return 45.0
    try:
        dob = datetime.strptime(dob_str[:8], "%Y%m%d")
        today = datetime.utcnow()
        age = (today - dob).days / 365.25
        return round(age, 1)
    except ValueError:
        return 45.0


def _encode_pos(pos: str) -> int:
    pos_map = {
        "11": 1, "21": 2, "22": 3, "23": 4,
        "24": 5, "31": 6, "81": 7,
    }
    return pos_map.get(pos, 0)
