from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional

import bcrypt
import jwt
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from .config import get_settings
from .database import get_collection

# MongoDB collection name for storing user data

USERS_COLLECTION = "users"

# Retrieve the users collection from MongoDB

def _get_users_collection() -> Collection:
    collection = get_collection(USERS_COLLECTION)
    if collection is None:
        raise RuntimeError("MongoDB connection is not configured. Set MONGO_URI and MONGO_DB_NAME.")
    return collection

# Hash a plaintext password using bcrypt

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

# Verify a plaintext password against its hashed version

def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        return False

# Register a new user in the database

def register_user(email: str, password: str, full_name: Optional[str] = None, plan: Optional[str] = None) -> Dict[str, Any]:
    email_normalised = email.strip().lower()
    plan_normalised = (plan or "free").strip().lower()
    collection = _get_users_collection()

    # Check if user already exists
    existing = collection.find_one({"email": email_normalised})
    if existing:
        raise ValueError("A user with this email already exists.")
    
    # Create user document with hashed password and metadata
    user_document = {
        "email": email_normalised,
        "password_hash": hash_password(password),
        "full_name": full_name,
        "plan": plan_normalised,
    "created_at": datetime.now(UTC),
    }

    result = collection.insert_one(user_document)
    return {
        "id": str(result.inserted_id),
        "email": email_normalised,
        "full_name": full_name,
        "plan": plan_normalised,
    }

# Authenticate user by verifying credentials

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    email_normalised = email.strip().lower()
    try:
        collection = _get_users_collection()
    except RuntimeError:
        return None

    user = collection.find_one({"email": email_normalised})
    if not user:
        return None

    password_hash = user.get("password_hash", "")
    if not verify_password(password, password_hash):
        return None

    # Prepare user object for return
    user_id = user.pop("_id", None)
    if user_id is not None:
        user["id"] = str(user_id)
    user.pop("password_hash", None)
    return user

# Create a JWT access token for the authenticated user

def create_access_token(user_id: str, email: str) -> str:
    settings = get_settings()
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET must be configured to issue tokens.")

    expires_delta = timedelta(minutes=settings.jwt_expires_minutes)
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + expires_delta,
    }

    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

# Decode a JWT token and return its payload

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    settings = get_settings()
    if not settings.jwt_secret:
        return None

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except jwt.PyJWTError:
        return None

# Retrieve the current user based on the provided JWT token

def get_current_user(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    payload = decode_token(token)
    if not payload:
        return None

    try:
        collection = _get_users_collection()
    except RuntimeError:
        return None

    try:
        user = collection.find_one({"email": payload.get("email")})
    except PyMongoError:
        return None

    if not user:
        return None

    user_id = user.pop("_id", None)
    if user_id is not None:
        user["id"] = str(user_id)
    user.pop("password_hash", None)
    return user

# Exported functions for external use

__all__ = [
    "register_user",
    "authenticate_user",
    "create_access_token",
    "decode_token",
    "get_current_user",
]
