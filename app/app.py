import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction", page_icon="🏦", layout="wide")

# Modern CSS styling
st.markdown("""
<style>
    /* Modern color palette */
    :root {
        --primary: #2563eb;
        --secondary: #7c3aed;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1e293b;
        --light: #f8fafc;
        --gray: #94a3b8;
        --tab-inactive: #e2e8f0;
        --tab-hover: #cbd5e1;
    }
    
    /* Main background and text colors */
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
        color: var(--dark);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark);
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 14px 28px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, var(--success) 0%, #06b6d4 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-size: 15px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Card styling */
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin: 20px 0;
        border-left: 6px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s ease;
        border-top: 4px solid var(--primary);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 15px 0;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-title {
        font-size: 1.1rem;
        color: var(--gray);
        font-weight: 600;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        font-weight: 600;
    }
    
    /* Input field styling */
    .stSlider, .stNumberInput, .stSelectbox {
        background: white;
        border-radius: 12px;
        padding: 5px;
    }
    
    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
    }
    
    /* Section dividers */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 30px 0;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    /* File uploader */
    .stFileUploader {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header banner
st.markdown("""
<div class="header-banner">
    <h1>📊 Customer Churn Prediction App</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Predict customer churn risk with machine learning</p>
</div>
""", unsafe_allow_html=True)

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
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("🔍 Predict Churn for a Single Customer")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", sorted([x.capitalize() for x in marital_le.classes_]))
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    occupation = st.selectbox("Occupation", sorted([x.capitalize() for x in occupation_le.classes_]))
    income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
    education = st.selectbox("Education Level", sorted([x.capitalize() for x in education_le.classes_]))

with col2:
    tenure_years = st.slider("Tenure (Years)", 0, 30, 5)
    segment = st.selectbox("Customer Segment", sorted([x.capitalize() for x in segment_le.classes_]))
    preferred_contact = st.selectbox("Preferred Contact", sorted([x.upper() for x in contact_le.classes_]))
    credit_score = st.slider("Credit Score", 300, 850, 650)
    credit_history_years = st.slider("Credit History (Years)", 0, 30, 5)

with col3:
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

if st.button("🔮 Predict Churn"):
    prediction = xgb_model.predict(input_df)[0]
    proba = xgb_model.predict_proba(input_df)[0][1]
    proba_percent = proba * 10000

    # Display results in metric cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Churn Probability</div>
            <div class="metric-value">{proba_percent:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Retention Probability</div>
            <div class="metric-value">{100 - proba_percent:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_level = "High" if proba_percent > 50 else "Medium" if proba_percent > 40 else "Low"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Risk Level</div>
            <div class="metric-value">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)

    if proba_percent > 50:
        st.error("⚠️ This customer is likely to churn.")
    elif 40 < proba_percent <= 50:
        st.warning("⚠️ This customer shows borderline churn risk.")
    else:
        st.success("✅ This customer is likely to stay.")

# ─────────────────────────────────────────────────────────────
# 📤 Batch Prediction
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.header("📤 Batch Churn Prediction")

st.markdown("""
<div style="background: white; padding: 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
    <h4>📋 Instructions:</h4>
    <ul>
        <li>Download the template CSV below</li>
        <li>Fill in customer data using the same column names</li>
        <li>Upload the completed file to generate churn predictions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

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
    df_result['Churn_Probability'] = (probabilities * 10000).round(2)

    risk_messages = []
    for proba_percent in df_result['Churn_Probability']:
        if proba_percent > 50:
            risk_messages.append("⚠️ Likely to churn")
        elif 40 < proba_percent <= 50:
            risk_messages.append("⚠️ Borderline churn risk")
        else:
            risk_messages.append("✅ Likely to stay")

    df_result['Risk_Comment'] = risk_messages
    df_result['Churn_Prediction'] = predictions

    st.success("✅ Predictions generated!")
    st.dataframe(df_result, use_container_width=True)

    # Download button
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

    # Recalculate churners based on probability threshold
    churn_threshold = 50
    churn_count = (df_result['Churn_Probability'] > churn_threshold).sum()
    total_count = len(df_result)
    churn_rate = churn_count / total_count if total_count > 0 else 0

    st.markdown(f"""
    <div class="metric-card" style="margin: 20px 0;">
        <div class="metric-title">Estimated Churn Rate</div>
        <div class="metric-value">{churn_rate:.2%}</div>
        <p style="color: var(--gray); margin-top: 10px;">{churn_count} likely churners out of {total_count}</p>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # 📊 Dashboard Visuals
    st.subheader("📊 Churn Probability Distribution")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_result['Churn_Probability'], bins=20, color='#2563eb', edgecolor='white', alpha=0.8)
    ax.set_xlabel("Churn Probability (%)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Customer Count", fontsize=12, fontweight='bold')
    ax.set_title("Distribution of Churn Risk", fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #94a3b8;">
    <p>Customer Churn Prediction App | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)