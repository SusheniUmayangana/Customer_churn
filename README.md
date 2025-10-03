# 🔍 Customer Churn Prediction – XGBoost Dashboard 

This project delivers a stakeholder-facing churn prediction system using XGBoost. It includes a reusable preprocessing pipeline, model training, batch and single prediction support, and an interactive Streamlit dashboard. Designed for clarity, reproducibility, and business impact.

---

## 📁 Project Structure

│

├── app/
│       └── app.py              # Streamlit dashboard for predictions
│
├── customer_churn/
│       ├── __init__.py         # Package marker for imports
│       ├── auth.py             # JWT helpers and credential management
│       ├── config.py           # Environment variable loader
│       ├── database.py         # MongoDB connection utilities
│       └── preprocessing.py    # Shared preprocessing pipeline
│
├── data/
│       ├── botswana_bank_customer_churn.csv    # Raw dataset
│       └── test_batch.csv                     # Sample batch input for testing
│
├── docs/
│       └── FDM_MLB_G16-SOW.pdf     # Project scope and documentation
│
├── model/
│       ├── preprocessor.joblib    # Persisted preprocessing pipeline
│       └── xgb_churn_model.pkl    # Trained XGBoost model
│
├── notebooks/
│       ├── FDM_mini_project.ipynb   # Exploratory analysis and pipeline overview
│       └── model_comparison.ipynb   # Model evaluation and selection
│
├── scripts/
│       └── train_model.py       # End-to-end training entry point
│
├── tests/
│       ├── conftest.py          # Test configuration for imports
│       ├── test_batch_prediction.py
│       ├── test_encoders.py
│       └── test_single_prediction.py
│
├── requirements.txt           # Dependencies
└── README.md                  # You're reading it!

└── .env.example               # Sample environment configuration for MongoDB/JWT

│

---

## ⚙️ Setup Instructions

1. **Create and activate a virtual environment (Windows PowerShell)**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment secrets**
   ```powershell
   copy .env.example .env
   ```
   Update `.env` with your MongoDB URI, target database name, and a strong `JWT_SECRET`. Ensure the MongoDB instance is reachable before launching the app.

4. **Train or retrain the model (optional)**
   ```powershell
   python scripts/train_model.py
   ```

5. **Run the dashboard**
   ```powershell
   python -m streamlit run app/app.py
   ```

6. **Run the automated tests**
   ```powershell
   pytest tests
   ```

---

## 📊 Features
- ✅ Shared preprocessing pipeline reused for training, batch, and single inference
- ✅ Streamlit dashboard with risk thresholds and downloadable results
- ✅ Batch prediction template with automatic handling of unfamiliar labels
- ✅ Persisted XGBoost model and preprocessing artifacts for reproducible scoring
- ✅ Automated regression tests covering preprocessing and inference flows

## 🚀 Future Enhancements
- RESTful API with FastAPI
- Docker packaging and cloud deployment
- Fairness-constrained model retraining
- ROI estimation and trend analysis

