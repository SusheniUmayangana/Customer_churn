import joblib

def test_encoder_classes():
    occupation_le = joblib.load("model/le_occupation.pkl")
    assert "other" in occupation_le.classes_, "'other' label missing from encoder"
    assert len(occupation_le.classes_) > 1, "Encoder has too few classes"