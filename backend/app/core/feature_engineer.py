"""Transform parsed claim JSON into numeric feature vectors for XGBoost.

Uses Redis for caching historical denial rates (TTL = 5 min).
Falls back to direct PostgreSQL queries if Redis is unavailable.
"""

from datetime import datetime

import hashlib
import re
import numpy as np
import pandas as pd
import structlog

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.repositories import remittance_repo, claim_repo

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
FEATURE_VERSION = "v3"

# Ordered feature names matching the trained model
FEATURE_NAMES = [
    # ── Existing (14) ──
    "total_charge",
    "charge_per_line",
    "service_line_count",
    "has_multiple_cpt",
    "dx_count",
    "modifier_missing",
    "patient_age",
    "place_of_service_encoded",
    "prior_auth_present",
    "payer_denial_rate",         # now Bayesian-smoothed
    "cpt_denial_rate",           # now Bayesian-smoothed
    "provider_denial_rate",      # now Bayesian-smoothed
    "invalid_npi",
    "duplicate_risk",
    # ── Aggregate confidence (3) ──
    "payer_denial_rate_n",
    "cpt_denial_rate_n",
    "provider_denial_rate_n",
    # ── Claim-level intelligence (10) ──
    "modifier_count",
    "dx_specificity",
    "cpt_category",
    "patient_gender_encoded",
    "has_rendering_provider",
    "taxonomy_category",
    "frequency_code_encoded",
    "payer_sequence_encoded",
    "filing_lag_days",
    "charge_dx_ratio",
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
    "invalid_npi": "Invalid NPI Format",
    "duplicate_risk": "Duplicate Claim Risk",
    "payer_denial_rate_n": "Payer Rate Sample Count",
    "cpt_denial_rate_n": "CPT Rate Sample Count",
    "provider_denial_rate_n": "Provider Rate Sample Count",
    "modifier_count": "Modifier Count",
    "dx_specificity": "Diagnosis Specificity",
    "cpt_category": "CPT Category",
    "patient_gender_encoded": "Patient Gender",
    "has_rendering_provider": "Has Rendering Provider",
    "taxonomy_category": "Provider Specialty Category",
    "frequency_code_encoded": "Claim Frequency Code",
    "payer_sequence_encoded": "Payer Sequence",
    "filing_lag_days": "Filing Lag (Days)",
    "charge_dx_ratio": "Charge to Diagnosis Ratio",
}

_NPI_RE = re.compile(r"^\d{10}$")


def _encode_cpt_category(cpt_code: str) -> int:
    """Encode CPT code into broad category."""
    if not cpt_code:
        return 0
    if cpt_code.startswith("99"):
        return 1  # E/M
    first = cpt_code[0] if cpt_code else ""
    if first in ("1", "2", "3", "4", "5", "6"):
        return 2  # Surgical
    if first == "7":
        return 3  # Radiology
    if first == "8":
        return 4  # Pathology/Lab
    if first == "9":
        return 5  # Medicine (non-E/M 9xxxx)
    return 0


def _encode_gender(gender: str) -> int:
    g = (gender or "").upper().strip()
    if g in ("M", "MALE"):
        return 1
    if g in ("F", "FEMALE"):
        return 2
    return 0


def _encode_taxonomy_category(taxonomy: str) -> int:
    """Encode provider taxonomy code into broad specialty category."""
    if not taxonomy:
        return 0
    prefix = taxonomy[:3] if len(taxonomy) >= 3 else taxonomy
    category_map = {
        "207": 1,   # Allopathic physicians
        "208": 2,   # Pediatrics, OB/GYN
        "261": 3,   # Ambulatory care
        "282": 4,   # Hospital
        "363": 5,   # Nursing
        "332": 6,   # Optometry
        "225": 7,   # Physical/Occupational therapy
        "174": 8,   # Pharmacist
        "367": 9,   # Technician
        "193": 10,  # Group practices
    }
    return category_map.get(prefix, 0)


def _encode_frequency_code(freq: str) -> int:
    mapping = {"1": 1, "6": 6, "7": 7, "8": 8}
    return mapping.get(str(freq or ""), 0)


def _encode_payer_sequence(seq: str) -> int:
    mapping = {"P": 1, "S": 2, "T": 3}
    return mapping.get((seq or "").upper()[:1], 0)


def _compute_filing_lag(service_lines: list[dict], created_at) -> float:
    """Days between earliest service date and claim creation."""
    dates = []
    for sl in service_lines:
        sd = sl.get("service_date", "")
        if sd and len(sd) >= 8:
            try:
                dates.append(datetime.strptime(sd[:8], "%Y%m%d"))
            except ValueError:
                pass
    if not dates:
        return 0.0
    earliest = min(dates)
    ref = created_at if isinstance(created_at, datetime) else datetime.utcnow()
    return max((ref - earliest).days, 0)


def _compute_dx_specificity(dx_codes: list[str]) -> float:
    """Average ICD-10 code length (proxy for diagnostic specificity)."""
    if not dx_codes:
        return 0.0
    lengths = [len(str(d).replace(".", "")) for d in dx_codes if d]
    return sum(lengths) / len(lengths) if lengths else 0.0


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

    # NPI format validation
    billing_npi = claim_doc.get("billing_provider_npi", "")
    rendering_npi = claim_doc.get("rendering_provider_npi", "")
    invalid_npi = 0
    if (billing_npi and not _NPI_RE.match(billing_npi)) or \
       (rendering_npi and not _NPI_RE.match(rendering_npi)):
        invalid_npi = 1

    # New claim-level features
    modifier_count = sum(
        len([m for m in sl.get("modifiers", []) if m])
        for sl in service_lines
    )
    dx_codes = claim_doc.get("diagnosis_codes", [])
    primary_cpt = service_lines[0].get("cpt_code", "") if service_lines else ""

    return {
        # Existing 14
        "total_charge": total_charge,
        "charge_per_line": total_charge / line_count,
        "service_line_count": line_count,
        "has_multiple_cpt": 1 if len(cpt_codes) > 1 else 0,
        "dx_count": len(dx_codes),
        "modifier_missing": modifier_missing,
        "patient_age": patient_age,
        "place_of_service_encoded": pos_encoded,
        "prior_auth_present": prior_auth,
        "payer_denial_rate": 0.0,
        "cpt_denial_rate": 0.0,
        "provider_denial_rate": 0.0,
        "invalid_npi": invalid_npi,
        "duplicate_risk": 0.0,  # enriched async in enrich_with_historical_rates
        # Aggregate confidence (enriched async, default 0)
        "payer_denial_rate_n": 0,
        "cpt_denial_rate_n": 0,
        "provider_denial_rate_n": 0,
        # New claim-level intelligence
        "modifier_count": modifier_count,
        "dx_specificity": _compute_dx_specificity(dx_codes),
        "cpt_category": _encode_cpt_category(primary_cpt),
        "patient_gender_encoded": _encode_gender(claim_doc.get("patient_gender", "")),
        "has_rendering_provider": 1 if (
            rendering_npi
            and rendering_npi != billing_npi
        ) else 0,
        "taxonomy_category": _encode_taxonomy_category(claim_doc.get("provider_taxonomy", "")),
        "frequency_code_encoded": _encode_frequency_code(claim_doc.get("frequency_code", "")),
        "payer_sequence_encoded": _encode_payer_sequence(claim_doc.get("payer_sequence", "")),
        "filing_lag_days": _compute_filing_lag(service_lines, claim_doc.get("created_at")),
        "charge_dx_ratio": total_charge / max(len(dx_codes), 1),
    }


async def enrich_with_historical_rates(session: AsyncSession, features: dict, claim_doc: dict) -> dict:
    """Enrich features with Bayesian-smoothed historical denial rates + sample counts."""
    payer_name = claim_doc.get("payer_name", "")
    payer_id = claim_doc.get("payer_id", "")
    if payer_id:
        cache_key = f"denial_rate:payer:{payer_name}"
        cached = await _get_cached_rate(cache_key)
        if cached is not None:
            features["payer_denial_rate"] = cached
        else:
            rate, count = await remittance_repo.get_denial_rate_with_count(session, "payer_name", payer_name)
            features["payer_denial_rate"] = rate
            features["payer_denial_rate_n"] = min(count, 500)
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
                rate, count = await remittance_repo.get_cpt_denial_rate_with_count(session, primary_cpt)
                features["cpt_denial_rate"] = rate
                features["cpt_denial_rate_n"] = min(count, 500)
                await _set_cached_rate(cache_key, rate)

    npi = claim_doc.get("billing_provider_npi", "")
    if npi:
        cache_key = f"denial_rate:provider:{npi}"
        cached = await _get_cached_rate(cache_key)
        if cached is not None:
            features["provider_denial_rate"] = cached
        else:
            rate, count = await remittance_repo.get_denial_rate_with_count(session, "payee_npi", npi)
            features["provider_denial_rate"] = rate
            features["provider_denial_rate_n"] = min(count, 500)
            await _set_cached_rate(cache_key, rate)

    # Duplicate risk: content-based duplicate detection
    subscriber_id = claim_doc.get("subscriber_id", "")
    billing_npi = claim_doc.get("billing_provider_npi", "")
    claim_id = claim_doc.get("claim_id", "")
    service_lines = claim_doc.get("service_lines", [])
    cpt_codes = sorted({sl.get("cpt_code", "") for sl in service_lines if sl.get("cpt_code")})
    service_dates = sorted({sl.get("service_date", "") for sl in service_lines if sl.get("service_date")})

    if subscriber_id and billing_npi and cpt_codes and service_dates:
        dup_cache_key = f"dup_risk:{subscriber_id}:{billing_npi}:{','.join(cpt_codes)}:{','.join(service_dates)}"
        cached = await _get_cached_rate(dup_cache_key)
        if cached is not None:
            features["duplicate_risk"] = cached
        else:
            count = await claim_repo.count_similar_claims(
                session, claim_id, subscriber_id, billing_npi, cpt_codes, service_dates,
            )
            dup_score = min(count, 3) / 3.0
            features["duplicate_risk"] = dup_score
            await _set_cached_rate(dup_cache_key, dup_score)

    return features


def features_to_dataframe(feature_dicts: list[dict]) -> pd.DataFrame:
    """Convert list of feature dicts to a DataFrame with ordered columns."""
    return pd.DataFrame(feature_dicts, columns=FEATURE_NAMES).fillna(0.0)


def align_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Align feature DataFrame to the loaded model's expected columns.

    Prevents XGBoost crash when current feature set (v2, 14 cols) is wider
    than the loaded model (v1, 12 cols).
    """
    from app.core.predictor import get_model_feature_count
    model_n = get_model_feature_count()
    if model_n is None or model_n == len(FEATURE_NAMES):
        return df  # compatible or unknown

    if model_n < len(FEATURE_NAMES):
        # Model trained on fewer features — slice to what it knows
        model_cols = FEATURE_NAMES[:model_n]
        logger.warning("Feature version mismatch: model expects %d features, code has %d. "
                       "Slicing to model columns. Retrain to use new features.",
                       model_n, len(FEATURE_NAMES))
        return df[model_cols]

    # Model expects more features than code provides — pad with zeros
    for i in range(len(FEATURE_NAMES), model_n):
        df[f"_pad_{i}"] = 0.0
    logger.warning("Model expects %d features but code has %d. Padding with zeros.",
                   model_n, len(FEATURE_NAMES))
    return df


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
