# 🔍 Customer Churn Prediction – XGBoost Pipeline (`dev/sushenii`)

This branch contains a modular, reproducible pipeline for predicting customer churn using XGBoost. It includes data preprocessing, model training, evaluation, and a Flask API for serving predictions. Designed for clarity, scalability, and business relevance.

---

## 📁 Project Structure


dev/sushenii/

├── app/

│       └── flask_api.py              # Flask app for serving predictions

├── data/

│     ├── botswana_bank_customer_churn.csv    # Raw dataset

│     └── final_feature.csv         # Preprocessed feature set

├── docs/

│   └── FDM_MLB_G16-SOW.pdf     # Project scope and documentation

├── model/

│   └── xgb_churn_model.pkl     # Trained XGBoost model

├── notebooks/

│   ├── FDM_mini_project.ipynb   # Exploratory analysis and pipeline overview

│   └── model_training.py       # Scripted training pipeline

├── requirements.txt           # Dependencies

└── README.md                     # You're reading it!


---

## ⚙️ Setup Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

2. **Run the Flask API**
   ```bash
   python app/flask_api.py

3. **Train the model**
   ```bash
   python notebooks/model_training.py
