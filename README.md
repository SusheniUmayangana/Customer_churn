# 🔍 Customer Churn Prediction – XGBoost Dashboard 

**Live Demo**: [Launch the Streamlit App](https://customerchurn-mxjnme2nlrphzcscndcm4h.streamlit.app/)

A secure, stakeholder-facing churn prediction system powered by XGBoost and Streamlit. This dashboard supports batch and single predictions, integrates MongoDB Atlas for user authentication, and includes a reproducible preprocessing pipeline. Designed for clarity, robustness, and business impact.


---

## 📘 Executive Summary

This project helps businesses identify customers at risk of churn using a machine learning model trained on Botswana banking data. It features:

- 🔐 Secure login via JWT and MongoDB
- 📊 Interactive dashboard for churn prediction
- 🧠 Reusable preprocessing pipeline and persisted model
- 🧪 Automated tests for reliability and reproducibility

Built for deployment on [Streamlit Cloud](https://streamlit.io/cloud), this app is modular, interpretable, and ready for stakeholder demos.

---


## 📁 Project Structure


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

│       ├── final_feature.csv # Preprocessed feature set

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

│       ├── model_training.py # Scripted training pipeline

│       └── model_comparison.ipynb   # Model evaluation and selection

│

├── scripts/

│       └── train_model.py       # End-to-end training entry point

│

├── tests/

│       ├── conftest.py          # Test configuration for imports

│       ├── test_batch_prediction.py # Batch prediction test script

│       ├── test_encoders.py  # Encoder validation script

│       └── test_single_prediction.py  # Single prediction test script

│

├── app.py              # Streamlit dashboard for predictions

├── requirements.txt           # Dependencies

├── README.md                  # You're reading it!

├── setup.sh                # Shell script

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
   python -m pytest tests/
   ```

---

## 📊 Features
- ✅ Shared preprocessing pipeline reused for training, batch, and single inference
- ✅ Streamlit dashboard with risk thresholds and downloadable results
- ✅ Batch prediction template with automatic handling of unfamiliar labels
- ✅ Persisted XGBoost model and preprocessing artifacts for reproducible scoring
- ✅ Automated regression tests covering preprocessing and inference flows

---

## 🙋‍♀️ Contributors

This project was developed as part of the **Year 3 Semester 2 – FDM Mini Project** by Group G16:

- **IT23256378** – Kalubowila K. S. U  
- **IT23209152** – Liyanage D. S  
- **IT23242418** – Bogahawatta M. O  
- **IT23232136** – Gunasekara P. A. H. I

Third-year undergraduates at SLIIT, specializing in Data Science and committed to building secure, interpretable, and stakeholder-ready AI solutions.

---

## ✅ Project Status

This dashboard is feature-complete and deployed. All core functionality—including secure login, batch/single prediction, and model reproducibility—is validated and ready for stakeholder demonstration.

