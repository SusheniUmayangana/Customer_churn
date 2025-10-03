import joblib
import pandas as pd


def test_single_prediction_output():
    model = joblib.load("model/rf_churn_model.pkl")

    encoders = {
        'marital_status': joblib.load("model/le_marital_status.pkl"),
        'education': joblib.load("model/le_education.pkl"),
        'occupation': joblib.load("model/le_occupation.pkl"),
        'segment': joblib.load("model/le_segment.pkl"),
        'preferred_contact': joblib.load("model/le_preferred_contact.pkl"),
    }

    sample = {
        'gender': [1],  # male
        'marital_status': ['married'],
        'dependents': [1],
        'occupation': ['production engineer'],
        'income': [75000],
        'education': ['bachelor'],
        'tenure_years': [7],
        'segment': ['sme'],
        'preferred_contact': ['email'],
        'credit_score': [680],
        'credit_history_years': [6],
        'outstanding_debt': [1200],
        'balance': [18000],
        'products_count': [3],
        'complaints_count': [0],
        'age': [38]
    }

    input_df = pd.DataFrame(sample)
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col].str.strip().str.lower())

    proba = model.predict_proba(input_df)[0][1]
    assert 0 <= proba <= 1, "Probability out of bounds"