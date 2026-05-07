from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rcm_demo"
    MODEL_DIR: str = str(Path(__file__).resolve().parent.parent / "models")
    REDIS_URL: str = "redis://localhost:6379"
    RISK_HIGH_THRESHOLD: float = 0.7
    RISK_MEDIUM_THRESHOLD: float = 0.3

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
