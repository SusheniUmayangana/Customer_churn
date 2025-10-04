from __future__ import annotations
import streamlit as st
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from a local .env file when present.
load_dotenv()

# Define a frozen dataclass to hold application settings

@dataclass(frozen=True)
class Settings:
    """Collection of configurable environment values used across the app."""

    mongo_uri: Optional[str]
    mongo_db_name: Optional[str]
    jwt_secret: Optional[str]
    jwt_algorithm: str
    jwt_expires_minutes: int
    environment: str

    @property
    def has_database(self) -> bool:
        return bool(self.mongo_uri and self.mongo_db_name)

    @property
    def has_jwt_config(self) -> bool:
        return bool(self.jwt_secret)

# Cache the settings object to avoid reloading secrets multiple times

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings derived from environment variables."""

    return Settings(
        mongo_uri=st.secrets["MONGO_URI"],
        mongo_db_name=st.secrets["MONGO_DB_NAME"],
        jwt_secret=st.secrets["JWT_SECRET"],
        jwt_algorithm=st.secrets.get("JWT_ALGORITHM", "HS256"),
        jwt_expires_minutes=int(st.secrets.get("JWT_EXPIRES_MINUTES", "60")),
        environment=st.secrets.get("APP_ENV", "development"),
    )

# Exported symbols for external use

__all__ = ["Settings", "get_settings"]
