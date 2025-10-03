import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# 📥 Load dataset
df = pd.read_csv("data/final_feature.csv", low_memory=False)

# 🧹 Drop non-informative ID columns
df.drop(['rownumber', 'customer_id', 'surname', 'first_name', 'address', 'phone'], axis=1, inplace=True)

# � Optional downsampling for quicker experimentation while respecting class balance
MAX_TRAIN_SAMPLES = int(os.environ.get('MAX_TRAIN_SAMPLES', 80000))
if len(df) > MAX_TRAIN_SAMPLES:
    print(f"🔁 Downsampling dataset from {len(df):,} to {MAX_TRAIN_SAMPLES:,} rows for faster training")
    per_class_cap = max(1, MAX_TRAIN_SAMPLES // df['churned'].nunique())
    balanced_splits = []
    for label, group in df.groupby('churned'):
        take = min(len(group), per_class_cap)
        balanced_splits.append(group.sample(n=take, random_state=42))
    df = pd.concat(balanced_splits).reset_index(drop=True)

# �🧼 Manual standardization of messy categories
education_map = {
    "bachelor": "bachelor",
    "bachelor's": "bachelor",
    "diploma": "diploma",
    "high school": "high school",
    "hs": "high school",
    "master": "master",
    "master's": "master"
}

occupation_map = {
    "accommodation manager": "accommodation manager",
    "accommodation manager¡": "accommodation manager",
    "accommodation managerá": "accommodation manager",
    "accountant, chartered": "accountant, chartered",
    "accountant, chartered certified": "accountant, chartered certified",
    "accountant, chartered certifie¡": "accountant, chartered certified"
}

# 🔄 Normalize and apply mappings
df['education'] = df['education'].str.strip().str.lower().map(education_map).fillna(df['education'].str.strip().str.lower())
df['occupation'] = df['occupation'].str.strip().str.lower().map(occupation_map).fillna(df['occupation'].str.strip().str.lower())

# 🔄 Encode categorical columns with 'other' fallback
label_encoders = {}
os.makedirs("model", exist_ok=True)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip().str.lower()
    known_labels = df[col].unique()
    df[col] = df[col].apply(lambda x: x if x in known_labels else 'other')  # fallback logic
    le = LabelEncoder()
    le.fit(list(known_labels) + ['other'])  # ensure 'other' is included
    df[col] = le.transform(df[col])
    label_encoders[col] = le
    joblib.dump(le, f"model/le_{col}.pkl")
    print(f"✅ Saved encoder for: {col} (includes 'other')")

# 🎯 Split features and target
X = df.drop('churned', axis=1)
y = df['churned']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧼 Handle missing values
X_train = X_train.ffill()

# ⚖️ Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf_config = {
    'n_estimators': 250,
    'max_depth': None,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'bootstrap': True,
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'n_jobs': -1,
}

print("⚙️ Training Random Forest with configuration:")
for key, value in rf_config.items():
    print(f"   • {key}: {value}")

best_model = RandomForestClassifier(**rf_config)
best_model.fit(X_resampled, y_resampled)

# 📊 Evaluate model
y_pred = best_model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

# 💾 Save model
joblib.dump(best_model, "model/rf_churn_model.pkl")
print("\n✅ Model saved to model/rf_churn_model.pkl")

# ✅ Accuracy check
print("\n📈 Train Accuracy:", accuracy_score(y_resampled, best_model.predict(X_resampled)))
print("📉 Test Accuracy:", accuracy_score(y_test, y_pred))

# 🔄 Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print("📊 Cross-validated ROC-AUC:", scores.mean())