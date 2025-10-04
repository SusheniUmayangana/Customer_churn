from __future__ import annotations

from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from .config import get_settings

# Global variable to hold the MongoDB client instance

_client: Optional[MongoClient] = None


def get_client() -> Optional[MongoClient]:
    """Initialise and cache a MongoDB client instance."""

    global _client
    settings = get_settings()
    if not settings.mongo_uri:
        return None

    # Create a new client only if one doesn't already exist

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


def fetch_user_predictions(user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    collection = get_collection("predictions")
    if collection is None:
        return []

    try:
        # Query the collection for user's predictions, sorted by creation time

        cursor = collection.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        documents: List[Dict[str, Any]] = []
        for document in cursor:
            mongo_id = document.get("_id")
            if mongo_id is not None:
                document["_id"] = str(mongo_id)
            documents.append(document)
        return documents
    except PyMongoError:
        return []


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None

# Exported functions for external use

__all__ = ["get_client", "get_database", "get_collection", "fetch_user_predictions", "close_client"]
