import joblib
import pandas as pd

def test_single_prediction_output():
    model = joblib.load("model/xgb_churn_model.pkl")
    input_df = pd.DataFrame({
        'gender': [1],
        'marital_status': [0],
        'dependents': [1],
        'occupation': [2],
        'income': [5000],
        'education': [1],
        'tenure_years': [5],
        'segment': [0],
        'preferred_contact': [1],
        'credit_score': [650],
        'credit_history_years': [5],
        'outstanding_debt': [1000],
        'balance': [5000],
        'products_count': [2],
        'complaints_count': [0],
        'age': [35]
    })
    proba = model.predict_proba(input_df)[0][1]
    assert 0 <= proba <= 1, "Probability out of bounds"