from __future__ import annotations

from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from .config import get_settings

_client: Optional[MongoClient] = None


def get_client() -> Optional[MongoClient]:
    """Initialise and cache a MongoDB client instance."""

    global _client
    settings = get_settings()
    if not settings.mongo_uri:
        return None

    if _client is None:
        try:
            _client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
            # Trigger a lightweight ping to validate credentials eagerly.
            _client.admin.command("ping")
        except PyMongoError:
            _client = None
            return None
    return _client


def get_database() -> Optional[Database]:
    """Return the configured MongoDB database or ``None`` when unavailable."""

    settings = get_settings()
    client = get_client()
    if client is None or not settings.mongo_db_name:
        return None
    return client[settings.mongo_db_name]


def get_collection(name: str) -> Optional[Collection]:
    """Convenience accessor for a named collection."""

    database = get_database()
    if database is None:
        return None
    return database[name]


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


__all__ = ["get_client", "get_database", "get_collection", "close_client"]
