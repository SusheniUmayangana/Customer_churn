import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")

# Load encoders
marital_le = joblib.load("model/le_marital_status.pkl")
education_le = joblib.load("model/le_education.pkl")
occupation_le = joblib.load("model/le_occupation.pkl")
segment_le = joblib.load("model/le_segment.pkl")
contact_le = joblib.load("model/le_preferred_contact.pkl")

encoders = {
    'marital_status': marital_le,
    'education': education_le,
    'occupation': occupation_le,
    'segment': segment_le,
    'preferred_contact': contact_le
}

# Load model
xgb_model = joblib.load("model/xgb_churn_model.pkl")

# ─────────────────────────────────────────────────────────────
# 🔍 Single Customer Prediction
st.header("🔍 Predict Churn for a Single Customer")

gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", sorted([x.capitalize() for x in marital_le.classes_]))
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
occupation = st.selectbox("Occupation", sorted([x.capitalize() for x in occupation_le.classes_]))
income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
education = st.selectbox("Education Level", sorted([x.capitalize() for x in education_le.classes_]))
tenure_years = st.slider("Tenure (Years)", 0, 30, 5)
segment = st.selectbox("Customer Segment", sorted([x.capitalize() for x in segment_le.classes_]))
preferred_contact = st.selectbox("Preferred Contact", sorted([x.upper() for x in contact_le.classes_]))
credit_score = st.slider("Credit Score", 300, 850, 650)
credit_history_years = st.slider("Credit History (Years)", 0, 30, 5)
outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
balance = st.number_input("Account Balance", min_value=0.0, value=5000.0)
products_count = st.slider("Products Count", 1, 10, 2)
complaints_count = st.slider("Complaints Count", 0, 5, 0)
age = st.slider("Age", 18, 100, 35)

input_df = pd.DataFrame({
    'gender': [1 if gender.lower() == "male" else 0],
    'marital_status': [marital_le.transform([marital_status.lower().strip()])[0]],
    'dependents': [1 if dependents.lower() == "yes" else 0],
    'occupation': [occupation_le.transform([occupation.lower().strip()])[0]],
    'income': [income],
    'education': [education_le.transform([education.lower().strip()])[0]],
    'tenure_years': [tenure_years],
    'segment': [segment_le.transform([segment.lower().strip()])[0]],
    'preferred_contact': [contact_le.transform([preferred_contact.lower().strip()])[0]],
    'credit_score': [credit_score],
    'credit_history_years': [credit_history_years],
    'outstanding_debt': [outstanding_debt],
    'balance': [balance],
    'products_count': [products_count],
    'complaints_count': [complaints_count],
    'age': [age]
})

if st.button("Predict Churn"):
    prediction = xgb_model.predict(input_df)[0]
    proba = xgb_model.predict_proba(input_df)[0][1]
    proba_percent = proba * 100

    st.write(f"🔢 Churn Probability: {proba_percent:.2f}%")

    if proba_percent > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    elif 0.4 < proba_percent <= 0.5:
        st.warning("⚠️ This customer shows borderline churn risk.")
    else:
        st.success("✅ This customer is likely to stay.")

# ─────────────────────────────────────────────────────────────
# 📤 Batch Prediction
st.header("📤 Batch Churn Prediction")

st.markdown("""
**Instructions:**
- Download the template CSV below
- Fill in customer data using the same column names
- Upload the completed file to generate churn predictions
""")

# Provide template CSV
template_df = pd.DataFrame(columns=[
    'gender', 'marital_status', 'dependents', 'occupation', 'income',
    'education', 'tenure_years', 'segment', 'preferred_contact',
    'credit_score', 'credit_history_years', 'outstanding_debt',
    'balance', 'products_count', 'complaints_count', 'age'
])

template_csv = template_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📄 Download Template CSV",
    data=template_csv,
    file_name="churn_template.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Upload completed customer data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validate columns
    required_cols = template_df.columns.tolist()
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()

    # Normalize text columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()

    # Map binary columns
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df['dependents'] = df['dependents'].map({'yes': 1, 'no': 0})

    # Map unknown labels to 'other' and log them
    mapped_summary = {}
    for col in encoders:
        known_classes = set(encoders[col].classes_)
        mapped_summary[col] = sorted(set(df[col].dropna()) - known_classes)
        df[col] = df[col].apply(lambda x: x if x in known_classes else 'other')
        df[col] = encoders[col].transform(df[col])

    # Show mapping summary
    with st.expander("🔍 Labels Mapped to 'Other'"):
        for col, unknowns in mapped_summary.items():
            if unknowns:
                st.write(f"**{col}**: {unknowns} → mapped to 'other'")

    # Convert numeric columns to proper types
    numeric_cols = [
        'income', 'tenure_years', 'credit_score', 'credit_history_years',
        'outstanding_debt', 'balance', 'products_count',
        'complaints_count', 'age'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols)

    # Predict churn
    predictions = xgb_model.predict(df)
    probabilities = xgb_model.predict_proba(df)[:, 1]

    # Combine results
    df_result = df.copy()
    df_result['Churn_Probability'] = (probabilities * 100).round(2)

    risk_messages = []
    for proba_percent in df_result['Churn_Probability']:
        if proba_percent > 0.5:
            risk_messages.append("⚠️ Likely to churn")
        elif 0.4 < proba_percent <= 0.5:
            risk_messages.append("⚠️ Borderline churn risk")
        else:
            risk_messages.append("✅ Likely to stay")

    df_result['Risk_Comment'] = risk_messages
    df_result['Churn_Prediction'] = predictions

    st.success("✅ Predictions generated!")
    st.dataframe(df_result)

    # Download button
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

    # Recalculate churners based on probability threshold
    churn_threshold = 0.5
    churn_count = (df_result['Churn_Probability'] > churn_threshold).sum()
    total_count = len(df_result)
    churn_rate = churn_count / total_count if total_count > 0 else 0

    st.metric("Estimated Churn Rate", f"{churn_rate:.2%}", delta=f"{churn_count} likely churners out of {total_count}")

    # ─────────────────────────────────────────────────────────────
    # 📊 Dashboard Visuals
    st.subheader("📊 Churn Probability Distribution")
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.hist(df_result['Churn_Probability'], bins=10, color='salmon', edgecolor='black')
    ax.set_xlabel("Churn Probability (%)")
    ax.set_ylabel("Customer Count")
    ax.set_title("Distribution of Churn Risk")
    plt.tight_layout()
    st.pyplot(fig)

   