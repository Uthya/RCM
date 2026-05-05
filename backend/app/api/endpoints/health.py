from fastapi import APIRouter

from app.core.predictor import is_model_loaded
from app.db.mongodb import get_db

router = APIRouter()


@router.get("/health")
async def health_check():
    db_connected = False
    try:
        db = get_db()
        await db.command("ping")
        db_connected = True
    except Exception:
        pass

    return {
        "status": "ok" if db_connected else "degraded",
        "model_loaded": is_model_loaded(),
        "db_connected": db_connected,
    }
