import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional

from customer_churn.preprocessing import ChurnPreprocessor


HIGH_RISK_THRESHOLD = 0.5
MEDIUM_RISK_THRESHOLD = 0.4


@st.cache_resource(show_spinner=False)
def load_artifacts():
    preprocessor = ChurnPreprocessor.load("model/preprocessor.joblib")
    model = joblib.load("model/xgb_churn_model.pkl")
    return preprocessor, model


def format_option(column: str, value: str) -> str:
    if column == "dependents" and value.isdigit():
        return value
    if value in {"other", "unknown"}:
        return "Unknown / Other"
    return value.replace("_", " ").title()


def select_from_encoder(label: str, column: str, default: Optional[str] = None) -> str:
    classes = list(PREPROCESSOR.encoders_[column].classes_)
    display_options = [format_option(column, item) for item in classes]
    index = classes.index(default) if default in classes else 0
    chosen_display = st.selectbox(label, display_options, index=index)
    return classes[display_options.index(chosen_display)]


def derive_risk_message(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "⚠️ Likely to churn"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "⚠️ Borderline churn risk"
    return "✅ Likely to stay"


def show_unknowns(metadata: Dict[str, List[str]]) -> None:
    if not metadata:
        return
    with st.expander("🔍 Labels mapped to 'other'"):
        for column, values in metadata.items():
            st.write(f"**{column}** → {values}")


st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")

PREPROCESSOR, MODEL = load_artifacts()
REQUIRED_COLUMNS = list(PREPROCESSOR.categorical_columns) + list(PREPROCESSOR.numeric_columns)


# ─────────────────────────────────────────────────────────────
# 🔍 Single Customer Prediction
st.header("🔍 Predict Churn for a Single Customer")

gender = select_from_encoder("Gender", "gender", default="female")
marital_status = select_from_encoder("Marital Status", "marital_status", default="married")
dependents = select_from_encoder("Number of Dependents", "dependents", default="0")
occupation = select_from_encoder("Occupation", "occupation")
education = select_from_encoder("Education Level", "education")
segment = select_from_encoder("Customer Segment", "segment")
preferred_contact = select_from_encoder("Preferred Contact", "preferred_contact")

income = st.number_input("Annual Income (USD)", min_value=0.0, value=60000.0, step=1000.0)
tenure_years = st.slider("Tenure (Years)", 0, 40, 5)
credit_score = st.slider("Credit Score", 300, 850, 650)
credit_history_years = st.slider("Credit History (Years)", 0, 40, 5)
outstanding_debt = st.number_input("Outstanding Debt (USD)", min_value=0.0, value=5000.0, step=500.0)
balance = st.number_input("Account Balance (USD)", min_value=0.0, value=10000.0, step=500.0)
products_count = st.slider("Products Count", 1, 10, 2)
complaints_count = st.slider("Complaints Count", 0, 10, 0)
age = st.slider("Age", 18, 100, 35)

single_record = pd.DataFrame(
    [
        {
            "gender": gender,
            "marital_status": marital_status,
            "dependents": dependents,
            "occupation": occupation,
            "income": income,
            "education": education,
            "tenure_years": tenure_years,
            "segment": segment,
            "preferred_contact": preferred_contact,
            "credit_score": credit_score,
            "credit_history_years": credit_history_years,
            "outstanding_debt": outstanding_debt,
            "balance": balance,
            "products_count": products_count,
            "complaints_count": complaints_count,
            "age": age,
        }
    ]
)

if st.button("Predict Churn"):
    processed_single, unknown_meta = PREPROCESSOR.transform(single_record, track_unknowns=True)
    show_unknowns(unknown_meta)

    probability = MODEL.predict_proba(processed_single)[0][1]
    probability_percent = probability * 100

    st.write(f"🔢 Churn Probability: {probability_percent:.2f}%")

    risk_message = derive_risk_message(probability)
    if probability >= HIGH_RISK_THRESHOLD:
        st.error(risk_message)
    elif probability >= MEDIUM_RISK_THRESHOLD:
        st.warning(risk_message)
    else:
        st.success(risk_message)


# ─────────────────────────────────────────────────────────────
# 📤 Batch Prediction
st.header("📤 Batch Churn Prediction")

st.markdown(
    """
**Instructions:**
- Download the template CSV below
- Fill in customer data using the same column names
- Upload the completed file to generate churn predictions
"""
)

template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
st.download_button(
    label="📄 Download Template CSV",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="churn_template.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Upload completed customer data CSV", type=["csv"])

if uploaded_file is not None:
    raw_batch = pd.read_csv(uploaded_file)
    raw_batch.columns = [col.strip().lower() for col in raw_batch.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in raw_batch.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    processed_batch, unknown_meta = PREPROCESSOR.transform(raw_batch, track_unknowns=True)
    show_unknowns(unknown_meta)

    probabilities = MODEL.predict_proba(processed_batch)[:, 1]
    predictions = MODEL.predict(processed_batch)

    results = raw_batch.copy()
    results["churn_probability"] = (probabilities * 100).round(2)
    results["churn_prediction"] = predictions
    results["risk_comment"] = [derive_risk_message(prob) for prob in probabilities]

    st.success("✅ Predictions generated!")
    st.dataframe(results)

    st.download_button(
        label="📥 Download Predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

    churn_count = (probabilities >= HIGH_RISK_THRESHOLD).sum()
    total_count = len(results)
    churn_rate = churn_count / total_count if total_count > 0 else 0
    st.metric(
        "Estimated Churn Rate",
        f"{churn_rate:.2%}",
        delta=f"{int(churn_count)} likely churners out of {total_count}",
    )

    st.subheader("📊 Churn Probability Distribution")
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.hist(results["churn_probability"], bins=10, color="salmon", edgecolor="black")
    ax.set_xlabel("Churn Probability (%)")
    ax.set_ylabel("Customer Count")
    ax.set_title("Distribution of Churn Risk")
    plt.tight_layout()
    st.pyplot(fig)


   