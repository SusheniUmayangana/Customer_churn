import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from customer_churn.preprocessing import ChurnPreprocessor


RAW_DATA_PATH = "data/botswana_bank_customer_churn.csv"
PREPROCESSOR_PATH = "model/preprocessor.joblib"
MODEL_PATH = "model/xgb_churn_model.pkl"


def main() -> None:
    os.makedirs("model", exist_ok=True)

    # 📥 Load raw dataset
    raw_df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    preprocessor = ChurnPreprocessor()

    # 🧼 Fit preprocessing pipeline
    processed_df = preprocessor.fit_transform(raw_df)
    preprocessor.save(PREPROCESSOR_PATH)
    print(f"✅ Saved fitted preprocessor to {PREPROCESSOR_PATH}")

    # 🎯 Split features and target
    X = processed_df.drop(columns=[preprocessor.target_column])
    y = processed_df[preprocessor.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ⚖️ Apply SMOTE for class balancing
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # 🔍 Hyperparameter tuning
    param_grid = {
        "max_depth": [3, 4],
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
    }

    grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss"),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        verbose=1,
    )

    grid.fit(X_resampled, y_resampled)
    best_model = grid.best_estimator_

    # 📊 Evaluate model
    y_pred = best_model.predict(X_test)
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n🏆 Best Parameters:", grid.best_params_)

    # 💾 Persist model
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

    # ✅ Accuracy check
    print("\n📈 Train Accuracy:", accuracy_score(y_resampled, best_model.predict(X_resampled)))
    print("📉 Test Accuracy:", accuracy_score(y_test, y_pred))

    # 🔄 Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")
    print("📊 Cross-validated Accuracy:", scores.mean())


if __name__ == "__main__":
    main()