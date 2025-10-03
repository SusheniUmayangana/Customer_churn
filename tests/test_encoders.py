from customer_churn.preprocessing import ChurnPreprocessor


def test_encoder_classes():
    preprocessor = ChurnPreprocessor.load("model/preprocessor.joblib")
    for column, encoder in preprocessor.encoders_.items():
        assert "other" in encoder.classes_, f"'other' label missing from encoder for {column}"
        assert len(encoder.classes_) > 1, f"Encoder for {column} has too few classes"