import pandas as pd
import joblib

def test_batch_prediction_pipeline():
    model = joblib.load("model/xgb_churn_model.pkl")
    df = pd.read_csv("data/test_batch.csv")

    # Normalize text columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()

    # Binary mappings
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df['dependents'] = df['dependents'].map({'yes': 1, 'no': 0})

    # Load encoders
    encoders = {
        'marital_status': joblib.load("model/le_marital_status.pkl"),
        'education': joblib.load("model/le_education.pkl"),
        'occupation': joblib.load("model/le_occupation.pkl"),
        'segment': joblib.load("model/le_segment.pkl"),
        'preferred_contact': joblib.load("model/le_preferred_contact.pkl")
    }

    # Map unknowns to 'other' and encode
    for col in encoders:
        known_classes = set(encoders[col].classes_)
        df[col] = df[col].apply(lambda x: x if x in known_classes else 'other')
        df[col] = encoders[col].transform(df[col])

    # Convert numeric columns
    numeric_cols = [
        'income', 'tenure_years', 'credit_score', 'credit_history_years',
        'outstanding_debt', 'balance', 'products_count',
        'complaints_count', 'age'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    # Final check: all columns must be numeric
    assert all(df.dtypes.apply(lambda x: x in ['int64', 'float64'])), "Non-numeric columns remain"

    # Predict
    predictions = model.predict(df)
    assert len(predictions) == len(df), "Mismatch in prediction count"