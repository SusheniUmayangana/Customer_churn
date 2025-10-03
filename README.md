# 🔍 Customer Churn Prediction – XGBoost Dashboard 

This project delivers a stakeholder-facing churn prediction system using XGBoost. It includes data preprocessing, model training, fairness evaluation, batch and single prediction support, and an interactive Streamlit dashboard. Designed for clarity, reproducibility, and business impact.

---

## 📁 Project Structure

│

├── app/

│       └── app.py              # Streamlit dashboard for predictions

│

├── data/

│     ├── botswana_bank_customer_churn.csv    # Raw dataset

│     └── final_feature.csv         # Preprocessed feature set

│     └── test_batch.csv         # Sample batch input for testing

│

├── docs/

│   └── FDM_MLB_G16-SOW.pdf     # Project scope and documentation

│

├── model/

│   └── xgb_churn_model.pkl             # Trained XGBoost model

│   ├── le_marital_status.pkl          # Label encoder for marital status

│   ├── le_dependents.pkl              # Label encoder for dependents

│   ├── le_occupation.pkl              # Label encoder for occupation

│   ├── le_preferred_contact.pkl       # Label encoder for contact method

│   └── le_segment.pkl                 # Label encoder for customer segment

│

├── notebooks/

│   ├── FDM_mini_project.ipynb   # Exploratory analysis and pipeline overview

│   └── model_training.py       # Scripted training pipeline

│   └── model_comparison.ipynb     # Model evaluation and selection

│

├── tests/

│   ├── test_batch_prediction.py       # Batch prediction test script

│   └── test_encoders.py               # Encoder validation script

│   └── test_single_prediction.py     # Single prediction test script

│

├── requirements.txt           # Dependencies

└── README.md                     # You're reading it!

│

---

## ⚙️ Setup Instructions

1. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Train the model**
   ```bash
   python notebooks/model_training.py
   
4. **Run the dashboard**
   ```bash
   python -m streamlit run app/app.py

5. **Run tests**
   ```bash
   pytest tests/

---

## 📊 Features
- ✅ Single and batch prediction support
- ✅ Fallback logic for unknown labels ("other" category)
- ✅ Churn probability with risk comments
- ✅ Segment-wise churn breakdown and KPIs
- ✅ Fairness evaluation using Fairlearn
- ✅ Formal testing with pytest
- ✅ Clean, reproducible pipeline and modular design

## 🚀 Future Enhancements
- RESTful API with FastAPI
- Docker packaging and cloud deployment
- Fairness-constrained model retraining
- ROI estimation and trend analysis

