import streamlit as st
import pandas as pd
import joblib

# 🔄 Load encoders
marital_le = joblib.load("model/le_marital_status.pkl")
education_le = joblib.load("model/le_education.pkl")
occupation_le = joblib.load("model/le_occupation.pkl")
segment_le = joblib.load("model/le_segment.pkl")
contact_le = joblib.load("model/le_preferred_contact.pkl")

# 📦 Load model
model = joblib.load("model/xgb_churn_model.pkl")

st.title("📊 Customer Churn Prediction App")

# 🧾 Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", sorted(set([x.capitalize() for x in marital_le.classes_])))
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
occupation = st.selectbox("Occupation", sorted(set([x.capitalize() for x in occupation_le.classes_])))
income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
education = st.selectbox("Education Level", sorted(set([x.capitalize() for x in education_le.classes_])))
tenure_years = st.slider("Tenure (Years)", 0, 30, 5)
segment = st.selectbox("Customer Segment", sorted(set([x.capitalize() for x in segment_le.classes_])))
preferred_contact = st.selectbox("Preferred Contact", sorted(set([x.upper() for x in contact_le.classes_])))
credit_score = st.slider("Credit Score", 300, 850, 650)
credit_history_years = st.slider("Credit History (Years)", 0, 30, 5)
outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1000.0)
balance = st.number_input("Account Balance", min_value=0.0, value=5000.0)
products_count = st.slider("Products Count", 1, 10, 2)
complaints_count = st.slider("Complaints Count", 0, 5, 0)
age = st.slider("Age", 18, 100, 35)

# 🔢 Normalize and encode inputs
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

# 🔍 Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    proba_percent = proba * 100

    st.write(f"🔢 Churn Probability: {proba_percent:.2f}%")

    if proba_percent > 1.0:
        st.error("⚠️ This customer is likely to churn.")
    elif 0.4 < proba_percent <= 1.0:
        st.warning("⚠️ This customer shows borderline churn risk.")
    else:
        st.success("✅ This customer is likely to stay.")