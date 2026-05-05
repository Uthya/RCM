from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.mongodb import connect_db, close_db, get_db
from app.core.predictor import load_model
from app.core.feature_engineer import init_cache
from app.api.router import api_router
from app.services.decision_engine import load_config as load_decision_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db()
    load_model()
    await load_decision_config(get_db())
    await init_cache()
    yield
    await close_db()


app = FastAPI(
    title="RCM AI Demo",
    description="Denial prediction pipeline for healthcare claims",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
