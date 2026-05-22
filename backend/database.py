import asyncio
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from backend.config import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None

# Index definitions per collection
INDEXES = {
    "commits": [
        IndexModel([("commit_hash", ASCENDING)], unique=True, name="commit_hash_unique"),
        IndexModel([("author_email", ASCENDING)], name="author_email"),
        IndexModel([("project_id", ASCENDING), ("created_at", DESCENDING)], name="project_created"),
        IndexModel([("created_at", DESCENDING)], name="created_at"),
    ],
    "developer_profiles": [
        IndexModel([("email", ASCENDING)], unique=True, name="email_unique"),
        IndexModel([("gitlab_username", ASCENDING)], sparse=True, name="gitlab_username"),
        IndexModel([("updated_at", DESCENDING)], name="updated_at"),
    ],
    "ghost_actions": [
        IndexModel([("action_type", ASCENDING), ("created_at", DESCENDING)], name="action_type_created"),
        IndexModel([("developer_id", ASCENDING), ("created_at", DESCENDING)], name="developer_created"),
        IndexModel([("status", ASCENDING)], name="status"),
        IndexModel([("created_at", DESCENDING)], name="created_at"),
    ],
}


async def connect(retries: int = 5, delay: float = 2.0) -> None:
    global _client, _db

    for attempt in range(1, retries + 1):
        try:
            _client = AsyncIOMotorClient(
                settings.mongodb_uri,
                maxPoolSize=settings.mongodb_max_pool_size,
                minPoolSize=settings.mongodb_min_pool_size,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
                retryWrites=True,
            )
            # Force connection on startup to catch config errors early
            await _client.admin.command("ping")
            _db = _client[settings.mongodb_db]
            await _create_indexes()
            logger.info("Connected to MongoDB at %s (db: %s)", settings.mongodb_uri, settings.mongodb_db)
            return
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            logger.warning("MongoDB connection attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                await asyncio.sleep(delay * attempt)  # exponential-ish back-off
            else:
                logger.error("Exhausted MongoDB connection retries")
                raise


async def disconnect() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
        logger.info("Disconnected from MongoDB")


async def _create_indexes() -> None:
    assert _db is not None
    for collection_name, index_models in INDEXES.items():
        collection = _db[collection_name]
        await collection.create_indexes(index_models)
        logger.debug("Indexes ensured for collection '%s'", collection_name)


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database is not connected. Call connect() first.")
    return _db


def get_collection(name: str):
    return get_db()[name]


async def ping() -> bool:
    """Return True if the database is reachable."""
    try:
        if _client is None:
            return False
        await _client.admin.command("ping")
        return True
    except Exception:
        return False
