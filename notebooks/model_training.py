import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 📥 Load dataset
df = pd.read_csv("data/final_feature.csv", low_memory=False)

# 🧹 Drop non-informative ID columns
df.drop(['rownumber', 'customer_id', 'surname', 'first_name', 'address', 'phone'], axis=1, inplace=True)

# 🔄 Encode categorical columns numerically
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 🎯 Split features and target
X = df.drop('churned', axis=1)
y = df['churned']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧼 Handle missing values
X_train = X_train.ffill()

# ⚖️ Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 🔍 Hyperparameter tuning with cross-validation
params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss'),
    param_grid=params,
    cv=5,
    scoring='f1',
    verbose=1
)

grid.fit(X_resampled, y_resampled)
best_model = grid.best_estimator_

# 📊 Evaluate best model
y_pred = best_model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n🧮 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\n🏆 Best Parameters:", grid.best_params_)

# 💾 Save best model
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/xgb_churn_model.pkl")
print("\n✅ Best model saved to model/xgb_churn_model.pkl")