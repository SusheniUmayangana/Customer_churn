# 🧪 Development Branch – XGBoost Pipeline (`dev/sushenii`)

This branch contains the active development pipeline for customer churn prediction using XGBoost. It includes experimental features, encoder validation, batch prediction upgrades, and Streamlit dashboard enhancements. All improvements here were tested, debugged, and later merged into main.

---

## 📁 Project Structure


dev/sushenii/

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

## 🧠 Development Highlights
- 🔄 Refactored batch prediction with template CSV and column validation
- 🧪 Validated encoders for unseen labels and added fallback logic ("other")
- 🧼 Improved error handling and user warnings in Streamlit
- 📊 Added churn probability interpretation and segment-wise KPIs
- ⚖️ Integrated fairness evaluation using Fairlearn
- 🧬 Compared multiple models and selected XGBoost for deployment
- 🧪 Formalized unit and integration tests using pytest

🚧 Status
This branch served as the foundation for the final production-ready version in main. All stable features have been merged. Future experimental work may continue here.
