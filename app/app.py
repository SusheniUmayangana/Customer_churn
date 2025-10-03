import os
from datetime import datetime
from itertools import cycle
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo.errors import PyMongoError

from customer_churn.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    register_user,
)
from customer_churn.database import get_collection
from customer_churn.preprocessing import ChurnPreprocessor

try:
    from weasyprint import HTML  # type: ignore
except ImportError:  # pragma: no cover
    HTML = None


HIGH_RISK_THRESHOLD = 0.5
MEDIUM_RISK_THRESHOLD = 0.4
AVERAGE_BALANCE_RISK_FACTOR = 0.12
DEFAULT_CUSTOMER_VALUE = 600.0

BRAND_ACCENT = "#114B8B"
BRAND_SECONDARY = "#F2B632"
BRAND_BACKGROUND = "linear-gradient(120deg, rgba(17,75,139,0.92), rgba(14,114,149,0.88))"

PREDICTIONS_COLLECTION = "predictions"
TOKEN_QUERY_PARAM = "auth_token"


def rerun_app() -> None:
    """Trigger a Streamlit rerun compatible with new and legacy APIs."""

    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:  # pragma: no cover - defensive guard for future API changes
        raise RuntimeError("Streamlit rerun functionality is unavailable; refresh the page manually.")


def _get_query_token() -> Optional[str]:
    if hasattr(st, "query_params"):
        value = st.query_params.get(TOKEN_QUERY_PARAM)
        if isinstance(value, list):
            return value[0] if value else None
        return value

    if not hasattr(st, "experimental_get_query_params"):
        return None

    params = st.experimental_get_query_params()
    value = params.get(TOKEN_QUERY_PARAM)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _set_query_token(token: Optional[str]) -> None:
    if hasattr(st, "query_params"):
        if token:
            st.query_params[TOKEN_QUERY_PARAM] = token
        elif TOKEN_QUERY_PARAM in st.query_params:
            del st.query_params[TOKEN_QUERY_PARAM]
        return

    if not hasattr(st, "experimental_get_query_params") or not hasattr(st, "experimental_set_query_params"):
        return

    params = st.experimental_get_query_params()
    if token:
        params[TOKEN_QUERY_PARAM] = [token]
    else:
        params.pop(TOKEN_QUERY_PARAM, None)
    st.experimental_set_query_params(**params)


def persist_auth_state(token: str, user: Dict[str, Any]) -> None:
    st.session_state["auth_token"] = token
    st.session_state["current_user"] = user
    _set_query_token(token)


def clear_auth_state() -> None:
    st.session_state["auth_token"] = None
    st.session_state["current_user"] = None
    _set_query_token(None)


FIELD_GROUPS: List[Dict[str, Any]] = [
    {
        "label": "Customer Profile",
        "columns": 2,
        "fields": [
            {
                "column": "gender",
                "type": "select",
                "label": "Gender",
                "default": "female",
                "help": "Primary gender on record",
            },
            {
                "column": "age",
                "type": "slider",
                "label": "Age",
                "min": 18,
                "max": 100,
                "default": 35,
                "step": 1,
                "help": "Customer age in years",
            },
            {
                "column": "marital_status",
                "type": "select",
                "label": "Marital Status",
                "default": "married",
            },
            {
                "column": "dependents",
                "type": "select",
                "label": "Number of Dependents",
                "default": "0",
            },
            {
                "column": "education",
                "type": "select",
                "label": "Education Level",
            },
            {
                "column": "occupation",
                "type": "select",
                "label": "Occupation",
            },
            {
                "column": "segment",
                "type": "select",
                "label": "Customer Segment",
            },
            {
                "column": "preferred_contact",
                "type": "select",
                "label": "Preferred Contact Method",
            },
        ],
    },
    {
        "label": "Account Engagement",
        "columns": 2,
        "fields": [
            {
                "column": "tenure_years",
                "type": "slider",
                "label": "Tenure (Years)",
                "min": 0,
                "max": 40,
                "default": 5,
                "step": 1,
            },
            {
                "column": "products_count",
                "type": "slider",
                "label": "Products Count",
                "min": 1,
                "max": 10,
                "default": 2,
                "step": 1,
            },
            {
                "column": "complaints_count",
                "type": "slider",
                "label": "Complaints Count",
                "min": 0,
                "max": 10,
                "default": 0,
                "step": 1,
            },
        ],
    },
    {
        "label": "Financial Snapshot",
        "columns": 2,
        "fields": [
            {
                "column": "income",
                "type": "number",
                "label": "Annual Income (USD)",
                "min": 0.0,
                "default": 60000.0,
                "step": 1000.0,
            },
            {
                "column": "balance",
                "type": "number",
                "label": "Account Balance (USD)",
                "min": 0.0,
                "default": 10000.0,
                "step": 500.0,
            },
            {
                "column": "outstanding_debt",
                "type": "number",
                "label": "Outstanding Debt (USD)",
                "min": 0.0,
                "default": 5000.0,
                "step": 500.0,
            },
        ],
    },
    {
        "label": "Credit History",
        "columns": 2,
        "fields": [
            {
                "column": "credit_score",
                "type": "slider",
                "label": "Credit Score",
                "min": 300,
                "max": 850,
                "default": 650,
                "step": 10,
            },
            {
                "column": "credit_history_years",
                "type": "slider",
                "label": "Credit History (Years)",
                "min": 0,
                "max": 40,
                "default": 5,
                "step": 1,
            },
        ],
    },
]


SEGMENT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Retail": {
        "gender": "female",
        "age": 32,
        "marital_status": "single",
        "dependents": "0",
        "education": "bachelor's",
        "occupation": "sales",
        "segment": "retail",
        "preferred_contact": "email",
        "tenure_years": 3,
        "products_count": 2,
        "complaints_count": 1,
        "income": 52000.0,
        "balance": 8500.0,
        "outstanding_debt": 4200.0,
        "credit_score": 640,
        "credit_history_years": 4,
    },
    "SME": {
        "gender": "male",
        "age": 45,
        "marital_status": "married",
        "dependents": "2",
        "education": "master's",
        "occupation": "entrepreneur",
        "segment": "sme",
        "preferred_contact": "phone",
        "tenure_years": 7,
        "products_count": 4,
        "complaints_count": 0,
        "income": 96000.0,
        "balance": 18500.0,
        "outstanding_debt": 7200.0,
        "credit_score": 690,
        "credit_history_years": 9,
    },
    "Corporate": {
        "gender": "female",
        "age": 51,
        "marital_status": "married",
        "dependents": "1",
        "education": "phd",
        "occupation": "executive",
        "segment": "corporate",
        "preferred_contact": "relationship_manager",
        "tenure_years": 11,
        "products_count": 6,
        "complaints_count": 0,
        "income": 168000.0,
        "balance": 42000.0,
        "outstanding_debt": 15000.0,
        "credit_score": 725,
        "credit_history_years": 15,
    },
}


SUGGESTION_RULES: List[Dict[str, Any]] = [
    {
        "title": "Offer refinance consultation",
        "description": "High outstanding debt with short credit history. Propose a refinancing package and credit coaching session.",
        "lift": "+18% retention uplift expected",
        "condition": lambda row: row.get("outstanding_debt", 0) > 8000 and row.get("credit_history_years", 0) < 5,
    },
    {
        "title": "Introduce loyalty rewards",
        "description": "Long-tenure client without complaints. Offering tiered rewards can pre-empt churn.",
        "lift": "+12% retention uplift expected",
        "condition": lambda row: row.get("tenure_years", 0) > 6 and row.get("complaints_count", 0) == 0,
    },
    {
        "title": "Schedule financial health check",
        "description": "Balance has been declining while product usage is low. Book a proactive review to upsell bundled products.",
        "lift": "+9% retention uplift expected",
        "condition": lambda row: row.get("balance", 0) < 7000 and row.get("products_count", 0) <= 2,
    },
    {
        "title": "Deploy relationship manager outreach",
        "description": "Preferred contact is phone and complaints were recorded. Have the RM follow up with a personalised retention plan.",
        "lift": "+15% retention uplift expected",
        "condition": lambda row: row.get("preferred_contact") == "phone" and row.get("complaints_count", 0) > 0,
    },
]


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
            .hero-container {{
                background: {BRAND_BACKGROUND};
                padding: 2.5rem;
                border-radius: 24px;
                color: white;
                margin-bottom: 1.5rem;
                position: relative;
                overflow: hidden;
                box-shadow: 0 18px 45px rgba(0,0,0,0.25);
            }}
            .hero-container::after {{
                content: "";
                position: absolute;
                top: -40%;
                right: -20%;
                width: 420px;
                height: 420px;
                background: rgba(255,255,255,0.15);
                border-radius: 50%;
                filter: blur(0.5px);
            }}
            .hero-title {{
                font-size: 2.1rem;
                font-weight: 700;
                margin-bottom: 0.75rem;
            }}
            .hero-subtitle {{
                font-size: 1.1rem;
                opacity: 0.9;
            }}
            .insight-card {{
                border-radius: 16px;
                padding: 1.25rem;
                background: rgba(255,255,255,0.98);
                border: 1px solid rgba(17,75,139,0.12);
                box-shadow: 0 10px 24px rgba(0,0,0,0.08);
                height: 100%;
            }}
            .insight-title {{
                font-weight: 600;
                font-size: 1.05rem;
                color: {BRAND_ACCENT};
            }}
            .insight-lift {{
                font-weight: 600;
                color: {BRAND_SECONDARY};
                margin-top: 0.65rem;
            }}
            div[data-testid="stMetricValue"] span {{
                font-size: 1.8rem;
            }}
            div[data-testid="stMetricDelta"] svg {{
                display: none;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Any:
    preprocessor = ChurnPreprocessor.load("model/preprocessor.joblib")
    model = joblib.load("model/xgb_churn_model.pkl")
    return preprocessor, model


def format_option(column: str, value: str) -> str:
    if column == "dependents" and value.isdigit():
        return value
    if value in {"other", "unknown"}:
        return "Unknown / Other"
    return value.replace("_", " ").title()


def select_from_encoder(
    label: str,
    column: str,
    default: Optional[str] = None,
    help_text: Optional[str] = None,
) -> str:
    classes = list(PREPROCESSOR.encoders_[column].classes_)
    display_options = [format_option(column, item) for item in classes]
    index = classes.index(default) if default in classes else 0
    chosen_display = st.selectbox(label, display_options, index=index, help=help_text)
    return classes[display_options.index(chosen_display)]


def render_field(field: Dict[str, Any], values: Dict[str, Any]) -> None:
    column_name = field["column"]
    label = field.get("label", column_name.replace("_", " ").title())
    help_text = field.get("help")
    field_type = field["type"]

    if field_type == "select":
        values[column_name] = select_from_encoder(
            label,
            column_name,
            default=field.get("default"),
            help_text=help_text,
        )
        return

    if field_type == "slider":
        slider_kwargs: Dict[str, Any] = {
            "min_value": field["min"],
            "max_value": field["max"],
            "value": field.get("default", field["min"]),
            "help": help_text,
        }
        if "step" in field:
            slider_kwargs["step"] = field["step"]
        values[column_name] = st.slider(label, **slider_kwargs)
        return

    if field_type == "number":
        default_value = field.get("default", field.get("min", 0.0))
        number_kwargs: Dict[str, Any] = {
            "min_value": field.get("min", 0.0),
            "value": default_value,
            "step": field.get("step", 1.0),
            "help": help_text,
        }
        if "max" in field:
            number_kwargs["max_value"] = field["max"]
        if "format" in field:
            number_kwargs["format"] = field["format"]
        values[column_name] = st.number_input(label, **number_kwargs)
        return

    values[column_name] = st.text_input(label, value=field.get("default", ""), help=help_text)


def build_default_values() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for group in FIELD_GROUPS:
        for field in group["fields"]:
            defaults[field["column"]] = field.get("default")
    return defaults


def get_segment_template(segment: str) -> Dict[str, Any]:
    template = SEGMENT_TEMPLATES.get(segment, {})
    base = build_default_values()
    base.update(template)
    return base


def derive_risk_message(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "⚠️ Likely to churn"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "⚠️ Borderline churn risk"
    return "✅ Likely to stay"


def categorize_risk(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def estimate_revenue_at_risk(df: pd.DataFrame) -> float:
    if "probability" not in df.columns:
        return 0.0
    high_risk = df[df["probability"] >= MEDIUM_RISK_THRESHOLD]
    if high_risk.empty:
        return 0.0
    if "balance" in high_risk.columns:
        return float(high_risk["balance"].fillna(0).sum() * AVERAGE_BALANCE_RISK_FACTOR)
    return float(len(high_risk) * DEFAULT_CUSTOMER_VALUE)


def reset_user_state() -> None:
    st.session_state["dashboard_stats"] = {
        "total_customers": 0,
        "total_high_risk": 0,
        "estimated_churn_rate": 0.0,
        "revenue_at_risk": 0.0,
    }
    st.session_state["client_notes"] = ""
    st.session_state["last_prediction"] = None
    st.session_state["batch_results"] = None
    st.session_state["last_results_for_report"] = None


def ensure_session_state() -> None:
    if "dashboard_stats" not in st.session_state:
        reset_user_state()
    else:
        st.session_state.setdefault("client_notes", "")
        st.session_state.setdefault("last_prediction", None)
        st.session_state.setdefault("batch_results", None)
        st.session_state.setdefault("last_results_for_report", None)

    st.session_state.setdefault("auth_token", None)
    st.session_state.setdefault("current_user", None)


def initialise_auth_state() -> None:
    token = st.session_state.get("auth_token")
    if not token:
        token = _get_query_token()
        if token:
            st.session_state["auth_token"] = token

    user = st.session_state.get("current_user")
    if token and not user:
        resolved_user = get_current_user(token)
        if resolved_user:
            st.session_state["current_user"] = resolved_user
        else:
            clear_auth_state()
            reset_user_state()


def sign_out() -> None:
    clear_auth_state()
    reset_user_state()
    rerun_app()


def render_account_sidebar() -> None:
    user = st.session_state.get("current_user")
    if not user:
        return

    with st.sidebar:
        display_name = user.get("full_name") or user.get("email", "")
        st.markdown(f"### 👋 {display_name}")
        st.caption(user.get("email", ""))
        st.divider()
        if st.button("Sign out", use_container_width=True):
            sign_out()


def render_authentication_gate() -> bool:
    """Render authentication tabs and return True when the user is logged in."""

    if st.session_state.get("current_user"):
        return True

    st.title("🔐 Secure access")
    st.caption("Sign in to unlock the retention command center")

    users_collection_available = get_collection("users") is not None
    if not users_collection_available:
        st.error(
            "MongoDB is not configured. Set MONGO_URI and MONGO_DB_NAME in your environment to enable authentication.",
        )
        return False

    login_column, register_column = st.columns(2)

    with login_column:
        st.subheader("Sign in")
        with st.form("login_form"):
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit_login = st.form_submit_button("Sign in", use_container_width=True)

        if submit_login:
            try:
                user = authenticate_user(email, password)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                if not user:
                    st.error("Invalid email or password.")
                else:
                    token = create_access_token(user["id"], user.get("email", ""))
                    persist_auth_state(token, user)
                    reset_user_state()
                    st.success("Welcome back! Redirecting...")
                    rerun_app()

    with register_column:
        st.subheader("Register")
        with st.form("register_form"):
            full_name = st.text_input("Full name", key="register_full_name")
            email = st.text_input("Corporate email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="register_confirm_password",
            )
            submit_register = st.form_submit_button("Create account", use_container_width=True)

        if submit_register:
            if password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters long.")
            else:
                try:
                    user = register_user(email, password, full_name=full_name or None)
                    token = create_access_token(user["id"], user.get("email", ""))
                except ValueError as exc:
                    st.error(str(exc))
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    persist_auth_state(token, user)
                    reset_user_state()
                    st.success("Account created successfully! Redirecting...")
                    rerun_app()

    st.info("Access is restricted to authorised bank personnel. Contact your administrator for assistance.")
    return False


def update_dashboard_stats(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return
    stats = st.session_state["dashboard_stats"]
    stats["total_customers"] += len(results_df)
    stats["total_high_risk"] += int((results_df["probability"] >= HIGH_RISK_THRESHOLD).sum())
    stats["revenue_at_risk"] += estimate_revenue_at_risk(results_df)
    if stats["total_customers"]:
        stats["estimated_churn_rate"] = stats["total_high_risk"] / stats["total_customers"]


def render_hero() -> None:
    stats = st.session_state["dashboard_stats"]
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-title">Banking Customer Retention Command Center</div>
            <div class="hero-subtitle">Real-time churn intelligence, tailored playbooks, and executive-ready reporting for relationship teams.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers Scored", f"{stats['total_customers']:,}")
    churn_rate = stats["estimated_churn_rate"]
    m2.metric("Estimated Churn Rate", f"{churn_rate:.1%}")
    m3.metric("Projected Revenue at Risk", f"${stats['revenue_at_risk']:,.0f}")


def show_unknowns(metadata: Dict[str, List[str]]) -> None:
    if not metadata:
        return
    with st.expander("🔍 Labels mapped to 'other'"):
        for column, values in metadata.items():
            st.write(f"**{column}** → {values}")


def generate_recommendations(row: Dict[str, Any], probability: float) -> List[Dict[str, str]]:
    recommendations: List[Dict[str, str]] = []
    for rule in SUGGESTION_RULES:
        try:
            if rule["condition"](row):
                recommendations.append(
                    {
                        "title": rule["title"],
                        "description": rule["description"],
                        "lift": rule["lift"],
                    }
                )
        except Exception:  # pragma: no cover
            continue
    if probability >= HIGH_RISK_THRESHOLD and not recommendations:
        recommendations.append(
            {
                "title": "Escalate to retention specialist",
                "description": "High churn likelihood detected. Assign a specialist to craft a bespoke retention bundle within 24 hours.",
                "lift": "+20% retention uplift expected",
            }
        )
    if probability < MEDIUM_RISK_THRESHOLD:
        recommendations.append(
            {
                "title": "Maintain proactive touchpoint",
                "description": "Low churn probability. Send a quarterly pulse survey and keep benefits reminders active.",
                "lift": "Maintains loyalty sentiment",
            }
        )
    return recommendations


def render_insight_cards(recommendations: List[Dict[str, str]]) -> None:
    if not recommendations:
        return
    st.subheader("🎯 Recommended retention actions")
    columns = st.columns(len(recommendations)) if len(recommendations) <= 3 else st.columns(3)
    col_cycle = cycle(columns)
    for rec in recommendations:
        with next(col_cycle):
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">{rec['title']}</div>
                    <div style="margin-top:0.75rem; line-height:1.45;">{rec['description']}</div>
                    <div class="insight-lift">{rec['lift']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_segment_charts(results: pd.DataFrame) -> None:
    if results.empty:
        return
    st.subheader("📈 Segment intelligence")
    results = results.copy()
    results["probability_percent"] = results["probability"] * 100

    col1, col2 = st.columns(2)
    with col1:
        fig_segment = px.bar(
            results,
            x="segment",
            y="probability_percent",
            color="risk_tier",
            barmode="group",
            labels={"probability_percent": "Avg churn probability (%)", "segment": "Customer segment", "risk_tier": "Risk tier"},
            title="Churn probability by segment",
        )
        fig_segment.update_layout(legend_title_text="Risk tier")
        st.plotly_chart(fig_segment, use_container_width=True)

    with col2:
        results["tenure_bucket"] = pd.cut(
            results["tenure_years"],
            bins=[-1, 2, 5, 10, 40],
            labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"],
        )
        fig_tenure = px.histogram(
            results,
            x="tenure_bucket",
            color="risk_tier",
            barmode="group",
            title="Churn risk by tenure cohort",
            labels={"tenure_bucket": "Tenure cohort", "count": "Customers"},
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    fig_scatter = px.scatter(
        results,
        x="complaints_count",
        y="probability_percent",
        color="risk_tier",
        size="balance" if "balance" in results.columns else None,
        labels={"complaints_count": "Complaints", "probability_percent": "Churn probability (%)"},
        title="Complaints vs churn probability",
        hover_data=["segment", "credit_score"],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def render_risk_tables(results: pd.DataFrame) -> None:
    if results.empty:
        return
    st.subheader("🧭 Risk tier playbooks")
    summary = (
        results.groupby("risk_tier")
        .agg(
            customers=("risk_tier", "count"),
            avg_probability=("probability", "mean"),
            avg_balance=("balance", "mean"),
            avg_credit_score=("credit_score", "mean"),
        )
        .reindex(["High", "Medium", "Low"])
        .fillna(0)
    )
    summary["avg_probability"] = summary["avg_probability"].apply(lambda v: f"{v*100:.1f}%")
    summary["avg_balance"] = summary["avg_balance"].apply(lambda v: f"${v:,.0f}")
    summary["avg_credit_score"] = summary["avg_credit_score"].apply(lambda v: f"{v:.0f}")

    playbooks = {
        "High": "Immediate outreach via relationship manager; assemble retention bundle within 24h.",
        "Medium": "Trigger nurture campaign; monitor credit score movements weekly.",
        "Low": "Maintain quarterly wellness checks and loyalty messaging.",
    }
    summary["recommended_playbook"] = [playbooks.get(idx, "") for idx in summary.index]

    styled = summary.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#0E7295"),
                    ("color", "white"),
                    ("text-align", "center"),
                ],
            }
        ]
    )
    st.dataframe(styled, use_container_width=True)


def log_single_prediction_entry(user: Dict[str, Any], inputs: Dict[str, Any], probability: float, risk_tier: str) -> None:
    collection = get_collection(PREDICTIONS_COLLECTION)
    if collection is None:
        return

    document = {
        "user_id": user.get("id"),
        "email": user.get("email"),
        "type": "single",
        "probability": float(probability),
        "risk_tier": risk_tier,
        "inputs": inputs,
        "created_at": datetime.utcnow(),
    }

    try:
        collection.insert_one(document)
    except PyMongoError:
        pass


def log_batch_prediction_entry(user: Dict[str, Any], results: pd.DataFrame) -> None:
    if results.empty:
        return

    collection = get_collection(PREDICTIONS_COLLECTION)
    if collection is None:
        return

    risk_counts = {tier: int(count) for tier, count in results["risk_tier"].value_counts().items()}
    summary = {
        "average_probability": float(results["probability"].mean()),
        "risk_counts": risk_counts,
        "records": int(len(results)),
    }
    sample_records = results.head(25).to_dict("records")

    document = {
        "user_id": user.get("id"),
        "email": user.get("email"),
        "type": "batch",
        "summary": summary,
        "sample": sample_records,
        "created_at": datetime.utcnow(),
    }

    try:
        collection.insert_one(document)
    except PyMongoError:
        pass


def get_model_metadata() -> Dict[str, str]:
    model_path = "model/xgb_churn_model.pkl"
    preprocessor_path = "model/preprocessor.joblib"
    metadata = {}
    if os.path.exists(model_path):
        metadata["Model artifact"] = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M")
    if os.path.exists(preprocessor_path):
        metadata["Preprocessor artifact"] = datetime.fromtimestamp(os.path.getmtime(preprocessor_path)).strftime("%Y-%m-%d %H:%M")
    metadata["Fairness assessment"] = "Refer to docs/model_comparison.ipynb"
    return metadata


def render_audit_panel() -> None:
    metadata = get_model_metadata()
    with st.expander("🛡️ Audit & compliance dossier", expanded=False):
        st.markdown("**Model provenance**")
        for key, value in metadata.items():
            st.write(f"- {key}: {value}")
        st.markdown("**Input validation rules**")
        st.write("- All categorical inputs constrained to learned encodings; unknowns mapped to `other` and flagged.")
        st.write("- Numerical features clipped to training distribution percentiles inside preprocessing pipeline.")
        st.write("- Batch uploads validated for required schema before scoring.")
        st.markdown("**Data sources**")
        st.write("- Botswana Bank transactional CRM export (2024).")
        st.write("- Complaints & engagement telemetry blended via preprocessing module.")
        note = "Install `weasyprint` for PDF exports or integrate directly with enterprise reporting tools (Power BI / Tableau)."
        st.info(note)


def build_report_html(results: pd.DataFrame) -> str:
    stats = st.session_state["dashboard_stats"]
    rows = results.to_dict("records")[:25]
    table_rows = "".join(
        f"<tr><td>{row.get('segment','')}</td><td>{row.get('risk_tier','')}</td><td>{row.get('probability',0)*100:.1f}%</td><td>${row.get('balance',0):,.0f}</td></tr>"
        for row in rows
    )
    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; color: #1b1b1b; }}
            h1 {{ color: {BRAND_ACCENT}; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background: {BRAND_ACCENT}; color: white; }}
        </style>
    </head>
    <body>
        <h1>Churn Retention Brief</h1>
        <p><strong>Generated:</strong> {datetime.now():%Y-%m-%d %H:%M}</p>
        <p><strong>Customers scored:</strong> {stats['total_customers']:,}</p>
        <p><strong>Estimated churn rate:</strong> {stats['estimated_churn_rate']:.1%}</p>
        <p><strong>Revenue at risk:</strong> ${stats['revenue_at_risk']:,.0f}</p>
        <table>
            <tr><th>Segment</th><th>Risk tier</th><th>Churn probability</th><th>Balance</th></tr>
            {table_rows}
        </table>
        <p style="margin-top: 24px;">Notes:<br/>{st.session_state.get('client_notes','')}</p>
    </body>
    </html>
    """


def render_user_management(latest_results: Optional[pd.DataFrame]) -> None:
    st.subheader("🗂️ Relationship manager workspace")
    st.session_state["client_notes"] = st.text_area(
        "Meeting notes / follow-up status",
        value=st.session_state.get("client_notes", ""),
        height=160,
    )

    if latest_results is None or latest_results.empty:
        st.info("Generate a prediction to enable PDF briefing.")
        return

    report_html = build_report_html(latest_results)
    if HTML:
        pdf_bytes = HTML(string=report_html).write_pdf()
        st.download_button(
            "📄 Generate PDF brief",
            data=pdf_bytes,
            file_name="churn_retention_brief.pdf",
            mime="application/pdf",
        )
    else:
        st.download_button(
            "📄 Download HTML brief",
            data=report_html.encode("utf-8"),
            file_name="churn_retention_brief.html",
            mime="text/html",
        )
        st.warning("Install `weasyprint` for direct PDF export: `pip install weasyprint`.")


def render_single_prediction_tab() -> pd.DataFrame:
    st.caption("Complete each section and submit to score an individual customer.")
    input_values: Dict[str, Any] = {}
    tabs = st.tabs([group["label"] for group in FIELD_GROUPS])
    for tab, group in zip(tabs, FIELD_GROUPS):
        with tab:
            columns = st.columns(group.get("columns", 2))
            column_cycle = cycle(columns)
            for field in group["fields"]:
                with next(column_cycle):
                    render_field(field, input_values)

    submitted = st.button("Predict Churn", type="primary")
    if not submitted:
        return pd.DataFrame()

    single_record = pd.DataFrame([input_values])
    processed_single, unknown_meta = PREPROCESSOR.transform(single_record, track_unknowns=True)
    show_unknowns(unknown_meta)

    probability = MODEL.predict_proba(processed_single)[0][1]
    probability_percent = probability * 100
    st.metric("Churn probability", f"{probability_percent:.2f}%", delta=None)
    risk_message = derive_risk_message(probability)
    if probability >= HIGH_RISK_THRESHOLD:
        st.error(risk_message)
    elif probability >= MEDIUM_RISK_THRESHOLD:
        st.warning(risk_message)
    else:
        st.success(risk_message)

    recommendations = generate_recommendations(single_record.iloc[0].to_dict(), probability)
    render_insight_cards(recommendations)

    results = single_record.copy()
    results["probability"] = probability
    results["risk_tier"] = results["probability"].apply(categorize_risk)
    user = st.session_state.get("current_user")
    if user:
        log_single_prediction_entry(
            user,
            single_record.iloc[0].to_dict(),
            probability,
            results["risk_tier"].iloc[0],
        )
    st.session_state["last_prediction"] = {
        "input": single_record.iloc[0].to_dict(),
        "probability": probability,
        "risk_tier": results["risk_tier"].iloc[0],
    }
    results_for_stats = results.copy()
    update_dashboard_stats(results_for_stats)
    st.session_state["last_results_for_report"] = results
    return results


def render_batch_tab(required_columns: List[str]) -> pd.DataFrame:
    st.markdown(
        """
        **Instructions**
        1. Download the template CSV
        2. Populate customer details using the provided schema
        3. Upload to generate predictions and executive dashboards
        """
    )
    template_df = pd.DataFrame(columns=required_columns)
    st.download_button(
        label="📄 Download Template CSV",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="churn_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload completed customer data CSV", type=["csv"])
    if uploaded_file is None:
        return pd.DataFrame()

    raw_batch = pd.read_csv(uploaded_file)
    raw_batch.columns = [col.strip().lower() for col in raw_batch.columns]
    missing = [col for col in required_columns if col not in raw_batch.columns]
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        return pd.DataFrame()

    processed_batch, unknown_meta = PREPROCESSOR.transform(raw_batch, track_unknowns=True)
    show_unknowns(unknown_meta)

    probabilities = MODEL.predict_proba(processed_batch)[:, 1]
    predictions = MODEL.predict(processed_batch)

    results = raw_batch.copy()
    results["probability"] = probabilities
    results["churn_probability"] = (probabilities * 100).round(2)
    results["prediction"] = predictions
    results["risk_comment"] = [derive_risk_message(prob) for prob in probabilities]
    results["risk_tier"] = results["probability"].apply(categorize_risk)

    st.success("✅ Predictions generated!")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        label="📥 Download Predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

    update_dashboard_stats(results)
    st.session_state["batch_results"] = results
    st.session_state["last_results_for_report"] = results
    user = st.session_state.get("current_user")
    if user:
        log_batch_prediction_entry(user, results)

    render_segment_charts(results)
    render_risk_tables(results)
    return results


def render_simulator_tab() -> None:
    st.markdown("Model alternative scenarios and stress-test retention strategies.")
    template_choice = st.selectbox("Persona template", options=list(SEGMENT_TEMPLATES.keys()))
    base_values = get_segment_template(template_choice)

    if st.session_state.get("last_prediction"):
        if st.checkbox("Start from last scored customer", value=False):
            base_values.update(st.session_state["last_prediction"]["input"])

    col1, col2, col3 = st.columns(3)
    base_values["tenure_years"] = col1.slider("Tenure (years)", 0, 40, int(base_values.get("tenure_years", 5)))
    base_values["credit_score"] = col2.slider("Credit score", 300, 850, int(base_values.get("credit_score", 650)))
    base_values["outstanding_debt"] = col3.number_input(
        "Outstanding debt (USD)",
        min_value=0.0,
        value=float(base_values.get("outstanding_debt", 5000.0)),
        step=500.0,
    )

    col4, col5 = st.columns(2)
    base_values["products_count"] = col4.slider("Products count", 1, 10, int(base_values.get("products_count", 2)))
    base_values["complaints_count"] = col5.slider("Complaints count", 0, 10, int(base_values.get("complaints_count", 0)))

    if st.button("Run scenario analysis"):
        scenario_df = pd.DataFrame([base_values])
        processed, unknown_meta = PREPROCESSOR.transform(scenario_df, track_unknowns=True)
        show_unknowns(unknown_meta)
        probability = MODEL.predict_proba(processed)[0][1]
        st.metric("Scenario churn probability", f"{probability*100:.2f}%")
        st.text(derive_risk_message(probability))
        recommendations = generate_recommendations(base_values, probability)
        render_insight_cards(recommendations)


def main() -> None:
    st.set_page_config(page_title="Bank Churn Intelligence", layout="wide")
    inject_global_styles()
    ensure_session_state()
    initialise_auth_state()

    if not render_authentication_gate():
        return

    render_account_sidebar()

    global PREPROCESSOR, MODEL
    PREPROCESSOR, MODEL = load_artifacts()
    required_columns = list(PREPROCESSOR.categorical_columns) + list(PREPROCESSOR.numeric_columns)

    render_hero()

    single_tab, batch_tab, simulator_tab = st.tabs([
        "Single Prediction",
        "Batch Insights",
        "What-if Simulator",
    ])

    with single_tab:
        latest_single = render_single_prediction_tab()

    with batch_tab:
        latest_batch = render_batch_tab(required_columns)

    with simulator_tab:
        render_simulator_tab()

    latest_results = st.session_state.get("last_results_for_report")
    render_user_management(latest_results if isinstance(latest_results, pd.DataFrame) else None)
    render_audit_panel()


if __name__ == "__main__":
    main()


