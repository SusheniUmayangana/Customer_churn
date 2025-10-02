import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# 📥 Load dataset
df = pd.read_csv("data/final_feature.csv", low_memory=False)

# 🧹 Drop non-informative ID columns
df.drop(['rownumber', 'customer_id', 'surname', 'first_name', 'address', 'phone'], axis=1, inplace=True)

# 🧼 Manual standardization of messy categories
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

# 🔄 Encode categorical columns and save encoders
label_encoders = {}
os.makedirs("model", exist_ok=True)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip().str.lower()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    joblib.dump(le, f"model/le_{col}.pkl")
    print(f"✅ Saved encoder for: {col}")

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

# 🔍 Hyperparameter tuning
params = {
    'max_depth': [3, 4],
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss'),
    param_grid=params,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    verbose=1
)

grid.fit(X_resampled, y_resampled)
best_model = grid.best_estimator_

# 📊 Evaluate model
y_pred = best_model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🏆 Best Parameters:", grid.best_params_)

# 💾 Save model
joblib.dump(best_model, "model/xgb_churn_model.pkl")
print("\n✅ Model saved to model/xgb_churn_model.pkl")

# ✅ Accuracy check
print("\n📈 Train Accuracy:", accuracy_score(y_resampled, best_model.predict(X_resampled)))
print("📉 Test Accuracy:", accuracy_score(y_test, y_pred))

# 🔄 Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
print("📊 Cross-validated Accuracy:", scores.mean())