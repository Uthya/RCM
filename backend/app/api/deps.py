"""Dependency injection for FastAPI endpoints."""

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.mongodb import get_db


async def get_database() -> AsyncIOMotorDatabase:
    return get_db()
