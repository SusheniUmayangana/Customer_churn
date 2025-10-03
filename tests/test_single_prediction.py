import joblib
import pandas as pd

from customer_churn.preprocessing import ChurnPreprocessor


def test_single_prediction_output():
    model = joblib.load("model/xgb_churn_model.pkl")
    preprocessor = ChurnPreprocessor.load("model/preprocessor.joblib")

    raw_input = pd.DataFrame(
        [
            {
                "gender": "male",
                "marital_status": "married",
                "dependents": "2",
                "occupation": "teacher",
                "income": 50000,
                "education": "bachelor",
                "tenure_years": 5,
                "segment": "sme",
                "preferred_contact": "phone",
                "credit_score": 650,
                "credit_history_years": 5,
                "outstanding_debt": 1000,
                "balance": 5000,
                "products_count": 2,
                "complaints_count": 0,
                "age": 40,
            }
        ]
    )

    processed, unknown_meta = preprocessor.transform(raw_input, track_unknowns=True)
    assert unknown_meta.get("occupation") == ["teacher"], unknown_meta

    proba = model.predict_proba(processed)[0][1]
    assert 0 <= proba <= 1, "Probability out of bounds"