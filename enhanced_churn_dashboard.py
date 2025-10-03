"""
Enhanced Customer Churn Prediction Dashboard
==========================================
Professional dashboard with improved UI/UX for customer churn prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)

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
    
    /* Card styling */
    .prediction-card {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin: 20px 0;
        border-left: 6px solid;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .churn-high {
        border-left-color: var(--danger);
    }
    
    .churn-medium {
        border-left-color: var(--warning);
    }
    
    .churn-low {
        border-left-color: var(--success);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease;
        border-top: 4px solid var(--primary);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
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
        min-height: 3rem;
    }
    
    .metric-title {
        font-size: 1.1rem;
        color: var(--gray);
        font-weight: 600;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--success), var(--primary));
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(160deg, var(--dark) 0%, #0f172a 100%);
        color: white;
        padding: 20px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Input field styling */
    .stSlider, .stNumberInput, .stSelectbox {
        background: white;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Risk indicator */
    .risk-indicator {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 700;
        color: white;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, var(--danger) 0%, #f97316 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, var(--warning) 0%, #eab308 100%);
    }
    
    .risk-low {
        background: linear-gradient(135deg, var(--success) 0%, #06b6d4 100%);
    }
    
    /* Enhanced Tab styling - Modern Button Style */
    .stTabs {
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        padding: 10px;
        margin-bottom: 25px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: none;
        padding: 5px;
        background: var(--light);
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        border-radius: 10px;
        font-weight: 600;
        color: var(--dark);
        background: var(--tab-inactive);
        border: none;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--tab-hover);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        color: white;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border: none;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
        transform: translateY(-2px);
    }
    
    /* Tab content styling */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 20px 5px;
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
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        color: var(--gray);
        font-size: 0.9rem;
        margin-top: 40px;
    }
    
    /* Fix for metric card content alignment */
    .metric-content {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model."""
    try:
        with open('target_range_random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("🚨 Model file not found. Please ensure 'target_range_random_forest_model.pkl' exists in the directory.")
        return None

def create_sidebar():
    """Create sidebar with app information."""
    st.sidebar.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=100&q=80", 
                     width=200)
    st.sidebar.title("🏦 ChurnGuard")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This Dashboard")
    st.sidebar.info("""
    This dashboard predicts customer churn probability using machine learning.
    
    **Key Features:**
    - Real-time predictions
    - Risk assessment
    - Actionable recommendations
    - Interactive visualizations
    """)
    
    st.sidebar.markdown("### How It Works")
    st.sidebar.info("""
    1. Enter customer details
    2. Click 'Predict Churn'
    3. View results and recommendations
    4. Take action to retain customers
    """)
    
    st.sidebar.markdown("### Model Performance")
    st.sidebar.success("""
    - **Accuracy:** 89.44%
    - **F1-Score:** 0.8439
    - **AUC-ROC:** 0.9583
    """)
    
    return st.sidebar

def create_input_features():
    """Create input features from user inputs with enhanced UI."""
    st.header("📋 Customer Profile")
    
    # Create tabs for better organization with modern button styling
    tab1, tab2, tab3 = st.tabs(["👤 Personal", "💰 Financial", "🏦 Banking"])
    
    with tab1:
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("🎂 Age", min_value=18, max_value=100, value=35, 
                           help="Customer's age in years")
            gender = st.selectbox("♂️ Gender", ["Male", "Female"],
                                help="Customer's gender")
            marital_status = st.selectbox("💍 Marital Status", 
                                        ["Single", "Married", "Divorced"],
                                        help="Customer's marital status")
            
        with col2:
            education = st.selectbox("🎓 Education Level", 
                                   ["High School", "Diploma", "Bachelor's", "Master's", "PhD"],
                                   help="Customer's highest education level")
            dependents = st.number_input("👶 Number of Dependents", min_value=0, max_value=10, value=0,
                                       help="Number of dependents the customer has")
    
    with tab2:
        st.subheader("Financial Information")
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.number_input("💼 Annual Income ($)", min_value=0, value=50000, step=1000,
                                   help="Customer's annual income in USD")
            balance = st.number_input("💰 Account Balance ($)", min_value=0, value=5000, step=100,
                                    help="Current account balance in USD")
            credit_score = st.slider("📈 Credit Score", min_value=300, max_value=850, value=650,
                                   help="Customer's credit score")
            
        with col2:
            outstanding_debt = st.number_input("💳 Outstanding Debt ($)", min_value=0, value=2000, step=100,
                                             help="Total outstanding debt in USD")
            credit_history_years = st.slider("📅 Credit History (Years)", min_value=0, max_value=50, value=5,
                                           help="Years of credit history")
    
    with tab3:
        st.subheader("Banking Details")
        col1, col2 = st.columns(2)
        
        with col1:
            tenure_years = st.slider("📆 Tenure with Bank (Years)", min_value=0, max_value=30, value=3,
                                   help="Years the customer has been with the bank")
            products_count = st.slider("📊 Number of Products", min_value=1, max_value=10, value=2,
                                     help="Number of banking products the customer uses")
            complaints_count = st.slider("📞 Number of Complaints", min_value=0, max_value=20, value=0,
                                       help="Number of complaints filed by the customer")
            
        with col2:
            segment = st.selectbox("🏢 Customer Segment", ["Retail", "SME", "Corporate"],
                                 help="Customer's banking segment")
            preferred_contact = st.selectbox("📱 Preferred Contact Method", 
                                           ["Email", "Phone", "Mail", "Text"],
                                           help="Customer's preferred communication method")
    
    # Create a dictionary with all features
    features = {
        'age': age,
        'tenure_years': tenure_years,
        'credit_score': credit_score,
        'credit_history_years': credit_history_years,
        'outstanding_debt': outstanding_debt,
        'Balance': balance,
        'products_count': products_count,
        'complaints_count': complaints_count,
        'Income': income,
        'credit_utilization': outstanding_debt / (income + 1),
        'clv': balance * tenure_years * 0.1,
        'risk_score': (
            (credit_score < 500) * 3 +
            (outstanding_debt / (income + 1) > 0.5) * 2 +
            (complaints_count > 3) * 2 +
            (products_count < 2) * 1
        ),
        'segment_encoded': {'Retail': 1, 'SME': 2, 'Corporate': 3}[segment],
        'education_encoded': {
            'High School': 1, 'Diploma': 2, 
            "Bachelor's": 3, "Master's": 4, 'PhD': 5
        }[education],
        'gender_encoded': 1 if gender == "Male" else 0,
        'marital_encoded': {'Single': 1, 'Married': 2, 'Divorced': 3}[marital_status],
        'contact_encoded': {'Email': 1, 'Phone': 2, 'Mail': 3, 'Text': 4}[preferred_contact]
    }
    
    return features

def preprocess_features(features):
    """Preprocess features to match training data format."""
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Ensure column order matches training data
    column_order = [
        'age', 'tenure_years', 'credit_score', 'credit_history_years',
        'outstanding_debt', 'Balance', 'products_count', 'complaints_count',
        'Income', 'credit_utilization', 'clv', 'risk_score',
        'segment_encoded', 'education_encoded', 'gender_encoded',
        'marital_encoded', 'contact_encoded'
    ]
    
    # Reorder columns
    df = df[column_order]
    
    return df

def make_prediction(model, features_df):
    """Make prediction using the model."""
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]
    
    return prediction, probability

def get_risk_level_and_color(churn_prob):
    """Determine risk level and color based on churn probability."""
    if churn_prob > 0.7:
        return "High", "#ef4444"
    elif churn_prob > 0.4:
        return "Medium", "#f59e0b"
    else:
        return "Low", "#10b981"

def create_gauge_chart(probability):
    """Create a gauge chart for churn probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if probability > 0.7 else "orange" if probability > 0.4 else "green"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}}))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'size': 16}
    )
    return fig

def create_feature_importance_chart():
    """Create feature importance visualization."""
    # Feature importance data (based on our model)
    features = ['Balance', 'CLV', 'Complaints', 'Products', 'Risk Score', 
                'Credit Score', 'Tenure', 'Income', 'Debt', 'Utilization']
    importance = [65.8, 16.6, 7.2, 4.1, 3.6, 2.1, 0.6, 0.01, 0.01, 0.01]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                 title="Feature Importance in Churn Prediction",
                 labels={'x': 'Importance (%)', 'y': 'Features'})
    
    fig.update_traces(
        marker_color=['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe', '#eff6ff'],
        marker_line_color='#1d4ed8',
        marker_line_width=1.5, 
        opacity=0.8
    )
    
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Importance (%)",
        yaxis_title="Features",
        title_font_size=20,
        title_x=0.5
    )
    return fig

def display_results(prediction, probability):
    """Display prediction results with enhanced UI."""
    st.header("🎯 Prediction Results")
    
    # Determine churn risk level - CONSISTENT THRESHOLDS
    churn_prob = probability[1]  # Probability of churn
    risk_level, risk_color = get_risk_level_and_color(churn_prob)
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-content">
                <div class="metric-title">Retention Probability</div>
                <div class="metric-value">{(1 - churn_prob):.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-content">
                <div class="metric-title">Churn Probability</div>
                <div class="metric-value">{churn_prob:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-content">
                <div class="metric-title">Risk Level</div>
                <div class="metric-value">{risk_level}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display prediction card with CONSISTENT risk level
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-card churn-high">
            <h2>🚨 CHURN ALERT</h2>
            <h3>This customer is likely to churn!</h3>
            <p><strong>Churn Probability: {churn_prob:.1%}</strong></p>
            <span class="risk-indicator risk-high">{risk_level.upper()} RISK</span>
            <p style="margin-top: 15px;"><strong>Recommended Action:</strong> Immediate personal outreach required</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-card churn-low">
            <h2>✅ ALL GOOD</h2>
            <h3>This customer is likely to stay!</h3>
            <p><strong>Churn Probability: {churn_prob:.1%}</strong></p>
            <span class="risk-indicator risk-low">{risk_level.upper()} RISK</span>
            <p style="margin-top: 15px;"><strong>Recommended Action:</strong> Standard relationship management</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create gauge chart
    st.plotly_chart(create_gauge_chart(churn_prob), use_container_width=True, key="gauge_chart")
    
    # Risk assessment
    st.subheader("📋 Risk Assessment")
    
    # Recommendations based on risk level
    st.subheader("💡 Recommendations")
    
    if churn_prob > 0.7:
        st.error("🚨 High Risk Customer - Immediate Action Required!")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Immediate Actions:
            - **Priority 1**: Immediate personal outreach within 24 hours
            - **Special Offers**: Consider exclusive retention offers
            - **Dedicated Support**: Assign a customer success manager
            """)
        with col2:
            st.markdown("""
            ### Investigation:
            - **Root Cause Analysis**: Investigate recent complaints or issues
            - **Competitor Check**: Determine if competitor offers are attracting the customer
            - **Value Proposition**: Reiterate unique bank benefits
            """)
    elif churn_prob > 0.4:
        st.warning("⚠️ Medium Risk Customer - Proactive Engagement Needed")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Engagement Strategies:
            - **Targeted Communication**: Send personalized retention messages
            - **Product Review**: Analyze product usage patterns
            - **Loyalty Programs**: Offer loyalty rewards or incentives
            """)
        with col2:
            st.markdown("""
            ### Relationship Building:
            - **Feedback Collection**: Proactively seek customer feedback
            - **Service Enhancement**: Address any service gaps
            - **Cross-selling**: Introduce complementary products
            """)
    else:
        st.success("✅ Low Risk Customer - Maintain Current Engagement")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Maintenance:
            - **Regular Check-ins**: Standard relationship management
            - **Upselling Opportunities**: Introduce new products or services
            - **Loyalty Rewards**: Continue with existing loyalty programs
            """)
        with col2:
            st.markdown("""
            ### Growth:
            - **Feedback Monitoring**: Keep monitoring for any changes
            - **Referral Program**: Encourage customer referrals
            - **Premium Services**: Introduce premium banking options
            """)

def main():
    """Main function for the enhanced Streamlit app."""
    # Create sidebar
    sidebar = create_sidebar()
    
    # Main header with modern banner
    st.markdown("""
    <div class="header-banner">
        <h1>🏦 Customer Churn Prediction Dashboard</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Predict customer churn in real-time with machine learning insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # App description
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 16px; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
        <h4>💡 Predict Customer Churn in Real-time</h4>
        <p>This dashboard uses machine learning to predict the likelihood of a customer leaving the bank. 
        Enter customer details to get instant predictions and actionable recommendations to improve retention.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Create input form
    features = create_input_features()
    
    # Predict button with enhanced styling
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 Predict Churn Probability", use_container_width=True):
            # Preprocess features
            features_df = preprocess_features(features)
            
            # Make prediction
            prediction, probability = make_prediction(model, features_df)
            
            # Display results
            display_results(prediction, probability)
            
            # Show input features for debugging
            with st.expander("🔍 See Input Features"):
                st.write("Input features used for prediction:")
                st.dataframe(features_df)
    
    # Feature importance section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("🧠 Model Insights")
    st.plotly_chart(create_feature_importance_chart(), use_container_width=True, key="feature_importance_insights")
    
    # App information
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("ℹ️ About This Model")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-content">
                <div class="metric-title">Model Type</div>
                <div class="metric-value">Random Forest</div>
                <p style="margin-top: 10px; color: #64748b;">Constrained for optimal performance</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-content">
                <div class="metric-title">Accuracy</div>
                <div class="metric-value">89.4%</div>
                <p style="margin-top: 10px; color: #64748b;">Within target range</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-content">
                <div class="metric-title">Last Updated</div>
                <div class="metric-value">Oct 2025</div>
                <p style="margin-top: 10px; color: #64748b;">Model version 1.2</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 30px; background: white; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.05);">
        <h4>🔒 Secure & Compliant</h4>
        <p>All data is processed locally and not stored. GDPR compliant.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Customer Churn Prediction Dashboard v2.0 | Powered by Machine Learning</p>
        <p>© 2025 ChurnGuard Analytics. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()