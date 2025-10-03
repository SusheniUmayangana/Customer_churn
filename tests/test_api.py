import os
import sys
from pathlib import Path

import joblib
import pytest

# Configure API credentials before importing the app factory
os.environ.setdefault("JWT_SECRET_KEY", "super-secret-key")
os.environ.setdefault("JWT_EXPIRES_IN_MINUTES", "5")
BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("API_USERNAME", "apitester")
os.environ.setdefault("API_PASSWORD", "secretpass")
os.environ.setdefault("JWT_SECRET_KEY", "super-secret-key")
os.environ.setdefault("JWT_EXPIRES_IN_MINUTES", "5")

from api.app import THRESHOLDS, create_app  # noqa: E402
from common.risk import classify_churn_risk  # noqa: E402


@pytest.fixture(scope="session")
def sample_record():
    """Build a representative payload using known encoder classes."""

    marital = joblib.load(BASE_DIR / "model" / "le_marital_status.pkl")
    education = joblib.load(BASE_DIR / "model" / "le_education.pkl")
    occupation = joblib.load(BASE_DIR / "model" / "le_occupation.pkl")
    segment = joblib.load(BASE_DIR / "model" / "le_segment.pkl")
    contact = joblib.load(BASE_DIR / "model" / "le_preferred_contact.pkl")

    return {
        "gender": "male",
        "marital_status": marital.classes_[0],
        "dependents": "yes",
        "occupation": occupation.classes_[0],
        "income": 5000,
        "education": education.classes_[0],
        "tenure_years": 5,
        "segment": segment.classes_[0],
        "preferred_contact": contact.classes_[0],
        "credit_score": 650,
        "credit_history_years": 5,
        "outstanding_debt": 1000,
        "balance": 5000,
        "products_count": 2,
        "complaints_count": 0,
        "age": 35,
    }


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture()
def token(client):
    response = client.post(
        "/auth/token",
        json={"username": os.environ["API_USERNAME"], "password": os.environ["API_PASSWORD"]},
    )
    assert response.status_code == 200
    return response.get_json()["access_token"]


def test_predict_requires_auth(client, sample_record):
    response = client.post("/predict", json=sample_record)
    assert response.status_code == 401


def test_single_prediction_endpoint(client, token, sample_record):
    response = client.post(
        "/predict",
        json=sample_record,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert set(payload.keys()) == {
        "prediction",
        "churn_probability",
        "churn_probability_percent",
        "risk_comment",
    }
    probability = payload["churn_probability"]
    assert 0 <= probability <= 1
    assert payload["risk_comment"] == classify_churn_risk(probability)


def test_batch_prediction_endpoint(client, token, sample_record):
    response = client.post(
        "/predict-batch",
        json={"records": [sample_record, sample_record]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["summary"]["total_records"] == 2
    assert len(payload["results"]) == 2
    for item in payload["results"]:
        assert 0 <= item["churn_probability"] <= 1
        assert item["risk_comment"] == classify_churn_risk(item["churn_probability"])


def test_risk_thresholds_alignment():
    assert classify_churn_risk(0.6) == "⚠️ Likely to churn"
    assert classify_churn_risk(0.45) == "⚠️ Borderline churn risk"
    assert classify_churn_risk(0.2) == "✅ Likely to stay"
    assert THRESHOLDS.high_risk == 0.5
    assert THRESHOLDS.borderline == 0.4
