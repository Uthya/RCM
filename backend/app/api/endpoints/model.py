"""Model management endpoints — retrain, status, training history."""

from fastapi import APIRouter
import structlog

from app.db.mongodb import get_db
from app.services.model_trainer import retrain_model, get_training_status

logger = structlog.get_logger()
router = APIRouter(prefix="/model", tags=["model"])


@router.get("/training-status")
async def training_status():
    """Get current training data availability and last training info."""
    db = get_db()
    return await get_training_status(db)


@router.post("/retrain")
async def retrain():
    """Retrain the model using real matched 837+835 data from MongoDB."""
    db = get_db()
    result = await retrain_model(db)
    return result
