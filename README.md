# Customer Churn Prediction Platform

An interactive churn-analytics workflow maintained by **Heshan**, covering data preparation, model training, and a Streamlit-based decision-support dashboard. The repository bundles everything needed to retrain the model, validate the outputs, and deploy a lightweight web app for stakeholders.

## Key Capabilities
- 🧭 **Exploratory analysis** notebooks for profiling churn behaviour.
- 🛠️ **Reproducible training script** (`notebooks/model_training.py`) that engineers features, tunes hyperparameters, and exports encoders/model artefacts.
- 🖥️ **Streamlit application** (`app/app.py`) for single-customer scoring, batch uploads, and probability visualisation.
- ✅ **Automated tests** (`tests/`) safeguarding encoder integrity, probability bounds, and batch prediction pipelines.

## Quickstart
```powershell
# create environment (example using venv)
python -m venv .venv
.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# launch the Streamlit app
streamlit run app/app.py
```

## Retraining the Model
```powershell
.venv\Scripts\Activate.ps1
python notebooks/model_training.py
```
The script regenerates feature encoders under `model/` and exports the updated classifier (`xgb_churn_model.pkl`).

## Project Layout
```
app/          Streamlit UI for churn insights
data/         Raw and feature-engineered datasets
docs/         Supporting documentation
model/        Serialized encoders and trained model
notebooks/    EDA, feature engineering, and training workflows
tests/        Pytest-based regression checks
```

## Maintainer
**Heshan** – data enthusiast focusing on customer retention analytics.
