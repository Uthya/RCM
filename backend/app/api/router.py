from fastapi import APIRouter

from app.api.endpoints import health, upload, claims, remittance, predict, dashboard, model

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(upload.router)
api_router.include_router(claims.router)
api_router.include_router(remittance.router)
api_router.include_router(predict.router)
api_router.include_router(dashboard.router)
api_router.include_router(model.router)
