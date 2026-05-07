from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.predictor import is_model_loaded
from app.api.deps import get_db

router = APIRouter()


@router.get("/health")
async def health_check(session: AsyncSession = Depends(get_db)):
    db_connected = False
    try:
        await session.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        pass

    return {
        "status": "ok" if db_connected else "degraded",
        "model_loaded": is_model_loaded(),
        "db_connected": db_connected,
    }
