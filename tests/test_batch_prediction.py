import joblib
import pandas as pd

from customer_churn.preprocessing import ChurnPreprocessor


def test_batch_prediction_pipeline():
    model = joblib.load("model/xgb_churn_model.pkl")
    preprocessor = ChurnPreprocessor.load("model/preprocessor.joblib")
    df = pd.read_csv("data/test_batch.csv")

    processed, unknown_meta = preprocessor.transform(df, track_unknowns=True)
    # Ensure any unknown labels are captured rather than silently ignored
    for column, values in unknown_meta.items():
        assert "other" in preprocessor.encoders_[column].classes_
        assert len(values) > 0

    assert processed.isnull().sum().sum() == 0, "Processed batch contains NaNs"

    predictions = model.predict(processed)
    assert len(predictions) == len(processed), "Mismatch in prediction count"