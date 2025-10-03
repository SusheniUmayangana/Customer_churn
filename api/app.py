from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import jwt
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from common.risk import THRESHOLDS, classify_churn_risk

# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = BASE_DIR / "model" / "xgb_churn_model.pkl"
ENCODER_PATHS = {
    "marital_status": BASE_DIR / "model" / "le_marital_status.pkl",
    "education": BASE_DIR / "model" / "le_education.pkl",
    "occupation": BASE_DIR / "model" / "le_occupation.pkl",
    "segment": BASE_DIR / "model" / "le_segment.pkl",
    "preferred_contact": BASE_DIR / "model" / "le_preferred_contact.pkl",
}

FEATURE_COLUMNS: List[str] = [
    "gender",
    "marital_status",
    "dependents",
    "occupation",
    "income",
    "education",
    "tenure_years",
    "segment",
    "preferred_contact",
    "credit_score",
    "credit_history_years",
    "outstanding_debt",
    "balance",
    "products_count",
    "complaints_count",
    "age",
]

NUMERIC_COLUMNS: Iterable[str] = (
    "income",
    "tenure_years",
    "credit_score",
    "credit_history_years",
    "outstanding_debt",
    "balance",
    "products_count",
    "complaints_count",
    "age",
)

MODEL = joblib.load(MODEL_PATH)
ENCODERS = {name: joblib.load(path) for name, path in ENCODER_PATHS.items()}

# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

JWT_ALGORITHM = "HS256"
DEFAULT_TOKEN_EXPIRY_MINUTES = 60


def _credentials() -> Dict[str, str]:
    """Fetch API credentials from environment with sensible defaults."""

    return {
        "username": os.getenv("API_USERNAME", "admin"),
        "password": os.getenv("API_PASSWORD", "admin123"),
        "secret": os.getenv("JWT_SECRET_KEY", "change-me"),
        "expires_in": int(
            os.getenv("JWT_EXPIRES_IN_MINUTES", DEFAULT_TOKEN_EXPIRY_MINUTES)
        ),
    }


def _generate_token(username: str) -> Dict[str, Any]:
    creds = _credentials()
    expiry = datetime.now(timezone.utc) + timedelta(minutes=creds["expires_in"])
    token = jwt.encode({"sub": username, "exp": expiry}, creds["secret"], JWT_ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": creds["expires_in"] * 60,
    }


def _decode_token(token: str) -> Dict[str, Any]:
    creds = _credentials()
    return jwt.decode(token, creds["secret"], algorithms=[JWT_ALGORITHM])


def _jwt_required(func):  # type: ignore[no-untyped-def]
    """Decorator enforcing bearer-token authentication."""

    @wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header missing"}), 401

        token = auth_header.split(" ", 1)[1]
        try:
            payload = _decode_token(token)
            request.user = payload.get("sub")  # type: ignore[attr-defined]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return func(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------


def _ensure_keys(record: Dict[str, Any]) -> None:
    missing = [col for col in FEATURE_COLUMNS if col not in record]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")


def _normalize_gender(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"male", "m"}:
            return 1
        if normalized in {"female", "f"}:
            return 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    raise ValueError("Gender must be 'male' or 'female'.")


def _normalize_dependents(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "y", "true", "1"}:
            return 1
        if normalized in {"no", "n", "false", "0"}:
            return 0
        if normalized.isdigit():
            return 1 if int(normalized) > 0 else 0
    if isinstance(value, (int, float)):
        return 1 if int(float(value)) > 0 else 0
    raise ValueError(
        "Dependents value must indicate whether dependents exist (yes/no or count)."
    )


def _encode_categorical(column: str, value: Any) -> int:
    encoder = ENCODERS[column]
    if not isinstance(value, str):
        raise ValueError(f"{column.replace('_', ' ').title()} must be provided as text.")

    normalized = value.strip().lower()
    if normalized not in encoder.classes_:
        normalized = "other" if "other" in encoder.classes_ else encoder.classes_[0]
    return int(encoder.transform([normalized])[0])


def _coerce_numeric(column: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - specific message
        raise ValueError(f"{column.replace('_', ' ').title()} must be numeric.") from exc


def _prepare_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    processed_rows: List[Dict[str, Any]] = []

    for index, raw_record in enumerate(records, start=1):
        _ensure_keys(raw_record)
        row: Dict[str, Any] = {}

        try:
            row["gender"] = _normalize_gender(raw_record["gender"])
            row["marital_status"] = _encode_categorical(
                "marital_status", raw_record["marital_status"]
            )
            row["dependents"] = _normalize_dependents(raw_record["dependents"])
            row["occupation"] = _encode_categorical("occupation", raw_record["occupation"])
            row["education"] = _encode_categorical("education", raw_record["education"])
            row["segment"] = _encode_categorical("segment", raw_record["segment"])
            row["preferred_contact"] = _encode_categorical(
                "preferred_contact", raw_record["preferred_contact"]
            )
        except ValueError as err:
            raise ValueError(f"Record {index}: {err}") from err

        # Numeric attributes
        for column in NUMERIC_COLUMNS:
            try:
                row[column] = _coerce_numeric(column, raw_record[column])
            except ValueError as err:
                raise ValueError(f"Record {index}: {err}") from err

        processed_rows.append(row)

    frame = pd.DataFrame(processed_rows, columns=FEATURE_COLUMNS)
    return frame


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health_check():  # type: ignore[no-untyped-def]
        return jsonify({"status": "ok", "model": MODEL_PATH.name})

    @app.post("/auth/token")
    def issue_token():  # type: ignore[no-untyped-def]
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", ""))
        password = str(payload.get("password", ""))
        creds = _credentials()

        if username != creds["username"] or password != creds["password"]:
            return jsonify({"error": "Invalid credentials"}), 401

        return jsonify(_generate_token(username))

    @app.post("/predict")
    @_jwt_required
    def predict_single():  # type: ignore[no-untyped-def]
        payload = request.get_json(silent=True) or {}
        try:
            frame = _prepare_dataframe([payload])
        except ValueError as err:
            return jsonify({"error": str(err)}), 400

        probabilities = MODEL.predict_proba(frame)[0][1]
        prediction = int(MODEL.predict(frame)[0])
        probability_value = float(probabilities)
        response = {
            "prediction": prediction,
            "churn_probability": probability_value,
            "churn_probability_percent": round(probability_value * 100, 2),
            "risk_comment": classify_churn_risk(probability_value),
        }
        return jsonify(response)

    @app.post("/predict-batch")
    @_jwt_required
    def predict_batch():  # type: ignore[no-untyped-def]
        payload = request.get_json(silent=True) or {}
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            return jsonify({"error": "'records' must be a non-empty list."}), 400

        try:
            frame = _prepare_dataframe(records)
        except ValueError as err:
            return jsonify({"error": str(err)}), 400

        probabilities = MODEL.predict_proba(frame)[:, 1]
        predictions = MODEL.predict(frame)

        results = []
        for raw, proba, pred in zip(records, probabilities, predictions):
            results.append(
                {
                    "input": raw,
                    "prediction": int(pred),
                    "churn_probability": float(proba),
                    "churn_probability_percent": round(float(proba) * 100, 2),
                    "risk_comment": classify_churn_risk(float(proba)),
                }
            )

        churn_rate = float((probabilities > THRESHOLDS.high_risk).mean())
        summary = {
            "total_records": len(results),
            "estimated_churn_rate": round(churn_rate, 4),
            "estimated_churn_rate_percent": round(churn_rate * 100, 2),
            "likely_churners": int((probabilities > THRESHOLDS.high_risk).sum()),
        }

        return jsonify({"summary": summary, "results": results})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
