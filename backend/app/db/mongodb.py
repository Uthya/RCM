import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings

logger = structlog.get_logger()

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None


async def connect_db() -> None:
    global _client, _db
    _client = AsyncIOMotorClient(settings.MONGO_URL)
    _db = _client[settings.MONGO_DB]
    # Create indexes
    await _db.claims.create_index("claim_id", unique=True)
    await _db.claims.create_index("payer_id")
    await _db.claims.create_index("created_at")
    await _db.remittances.create_index("claim_id")
    await _db.remittances.create_index("payer_control_number")
    await _db.remittances.create_index([("payer_name", 1), ("claim_status", 1)])
    await _db.remittances.create_index("payee_npi")
    await _db.predictions.create_index("claim_id", unique=True)
    await _db.predictions.create_index("risk_level")
    await _db.claims.create_index("actual_outcome")
    await _db.upload_history.create_index("uploaded_at")
    # Knowledge layer indexes
    await _db.fix_history.create_index([("payer_name", 1), ("issue_type", 1)])
    await _db.fix_history.create_index("cpt_code")
    logger.info("Connected to MongoDB", db=settings.MONGO_DB)


async def close_db() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
    logger.info("Closed MongoDB connection")


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialized. Call connect_db() first.")
    return _db
