from pathlib import Path
import sys
import time
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.features import create_features
from src.risk_engine import make_risk_decision


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ForensIQ AI",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = None


# ============================================================
# THEME
# ============================================================

theme = st.session_state.theme

if theme == "Light":

    BG = "#F6F8FB"
    CARD = "#FFFFFF"
    TEXT = "#172033"
    MUTED = "#687386"
    BORDER = "#E1E6EF"
    PRIMARY = "#2446A8"
    SIDEBAR = "#FFFFFF"
    INPUT_BG = "#FFFFFF"

else:

    BG = "#0E1117"
    CARD = "#171B24"
    TEXT = "#F4F7FB"
    MUTED = "#9AA4B2"
    BORDER = "#2B3342"
    PRIMARY = "#6C8CFF"
    SIDEBAR = "#11151D"
    INPUT_BG = "#1B202B"


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
    <style>

    html, body, [class*="css"] {{
        font-family: "Segoe UI", Arial, sans-serif;
    }}

    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    section[data-testid="stSidebar"] {{
        background: {SIDEBAR};
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] > div {{
        padding-top: 1.5rem;
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4 {{
        color: {TEXT} !important;
    }}

    p, label {{
        color: {TEXT};
    }}

    .forensiq-logo {{
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -2px;
        color: {PRIMARY};
        margin-bottom: 0;
    }}

    .forensiq-subtitle {{
        color: {MUTED};
        font-size: 0.95rem;
        margin-top: -8px;
        margin-bottom: 2rem;
    }}

    .section-title {{
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1.4rem;
        margin-bottom: 0.8rem;
        color: {TEXT};
    }}

    .info-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }}

    .status-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }}

    .status-dot {{
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #20A464;
        margin-right: 8px;
    }}

    .risk-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}

    .risk-high {{
        border-left: 6px solid #C0392B;
    }}

    .risk-medium {{
        border-left: 6px solid #D68910;
    }}

    .risk-low {{
        border-left: 6px solid #208A5A;
    }}

    .risk-title {{
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }}

    .risk-description {{
        color: {MUTED};
        font-size: 0.95rem;
    }}

    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem;
        min-height: 110px;
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 0.82rem;
        margin-bottom: 0.4rem;
    }}

    .metric-value {{
        color: {TEXT};
        font-size: 1.2rem;
        font-weight: 700;
        word-break: break-word;
    }}

    .reason-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        color: {TEXT};
    }}

    .scenario-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }}

    .footer {{
        text-align: center;
        color: {MUTED};
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid {BORDER};
    }}

    div[data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        padding: 1rem;
        border-radius: 12px;
    }}

    div[data-baseweb="input"],
    div[data-baseweb="select"] {{
        background: {INPUT_BG};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = PROJECT_ROOT / "models" / "forensiq_model.joblib"

    if not model_path.exists():
        return None

    artifact = joblib.load(model_path)

    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]

    return artifact


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        f"""
        <div class="forensiq-logo">ForensIQ</div>
        <div class="forensiq-subtitle">
            Explainable AI for transaction risk intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### System")

    if model is not None:
        st.success("AI model loaded")
    else:
        st.error("Model not found")

    st.divider()

    st.markdown("### Interface")

    selected_theme = st.radio(
        "Appearance",
        ["Light", "Dark"],
        index=0 if theme == "Light" else 1,
        key="appearance_selector"
    )

    if selected_theme != st.session_state.theme:

        st.session_state.theme = selected_theme

        st.rerun()

    st.divider()

    st.markdown("### About ForensIQ")

    st.caption(
        "ForensIQ evaluates transaction behaviour using engineered mathematical "
        "signals and machine-learning risk prediction."
    )

    st.caption(
        "The system provides a fraud probability, risk classification, "
        "recommended action, and an explanation of the strongest contributing factors."
    )

    st.divider()

    st.caption("Prototype environment")
    st.caption("Model: LightGBM")
    st.caption("Explainability: Feature contribution analysis")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="forensiq-logo">ForensIQ</div>
    <div class="forensiq-subtitle">
        Explainable AI for transaction fraud detection
    </div>
    """,
    unsafe_allow_html=True
)

st.success("System ready for transaction analysis")


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.markdown(
    '<div class="section-title">Transaction Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# SENDER
# ------------------------------------------------------------

with col1:

    st.markdown("#### Sender Account")

    name_orig = st.text_input(
        "Sender Account ID",
        value="C100000001",
        key="sender_account_id"
    )

    oldbalance_org = st.number_input(
        "Balance Before Transaction",
        min_value=0.0,
        value=500000.0,
        step=100.0,
        key="sender_balance_before"
    )

    newbalance_orig = st.number_input(
        "Balance After Transaction",
        min_value=0.0,
        value=0.0,
        step=100.0,
        key="sender_balance_after"
    )


# ------------------------------------------------------------
# TRANSACTION
# ------------------------------------------------------------

with col2:

    st.markdown("#### Transaction")

    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "TRANSFER",
            "CASH_OUT",
            "PAYMENT",
            "CASH_IN",
            "DEBIT"
        ],
        key="transaction_type"
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=500000.0,
        step=100.0,
        key="transaction_amount"
    )

    step = st.number_input(
        "Transaction Time Step",
        min_value=1,
        value=100,
        step=1,
        key="transaction_step"
    )


# ------------------------------------------------------------
# RECIPIENT
# ------------------------------------------------------------

with col3:

    st.markdown("#### Recipient Account")

    name_dest = st.text_input(
        "Recipient Account ID",
        value="C200000001",
        key="recipient_account_id"
    )

    oldbalance_dest = st.number_input(
        "Balance Before Transaction",
        min_value=0.0,
        value=0.0,
        step=100.0,
        key="recipient_balance_before"
    )

    newbalance_dest = st.number_input(
        "Balance After Transaction",
        min_value=0.0,
        value=0.0,
        step=100.0,
        key="recipient_balance_after"
    )


# ============================================================
# ANALYSIS BUTTON
# ============================================================

st.divider()

analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 1, 1])

with analyze_col2:

    analyze_button = st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
        key="analyze_transaction_button"
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if model is None:

        st.error(
            "The trained model could not be found. "
            "Please verify that models/forensiq_model.joblib exists."
        )

        st.stop()

    # --------------------------------------------------------
    # TRANSACTION TIMING
    # --------------------------------------------------------

    analysis_delay = random.uniform(1.0, 5.0)

    with st.spinner("ForensIQ is analysing transaction behaviour..."):

        time.sleep(analysis_delay)

        transaction_timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # --------------------------------------------------------
    # RAW TRANSACTION
    # --------------------------------------------------------

    transaction = pd.DataFrame(
        [
            {
                "step": step,
                "type": transaction_type,
                "amount": amount,
                "nameOrig": name_orig,
                "oldbalanceOrg": oldbalance_org,
                "newbalanceOrig": newbalance_orig,
                "nameDest": name_dest,
                "oldbalanceDest": oldbalance_dest,
                "newbalanceDest": newbalance_dest,
                "isFraud": 0,
                "isFlaggedFraud": 0
            }
        ]
    )

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    try:

        engineered = create_features(transaction)

    except Exception as error:

        st.error(f"Feature engineering failed: {error}")

        st.stop()

    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    drop_columns = [
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest"
    ]

    X = engineered.drop(
        columns=[
            column
            for column in drop_columns
            if column in engineered.columns
        ]
    )

    X = pd.get_dummies(
        X,
        columns=["type"]
    )

    # Match model feature names if available

    try:

        if hasattr(model, "feature_name_"):

            expected_features = model.feature_name_

            X = X.reindex(
                columns=expected_features,
                fill_value=0
            )

    except Exception:

        pass

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        fraud_probability = float(
            model.predict_proba(X)[0][1]
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # RISK DECISION
    # --------------------------------------------------------

    decision = make_risk_decision(
        fraud_probability
    )

    risk_level = decision["risk_level"]

    recommended_action = decision[
        "recommended_action"
    ]

    # --------------------------------------------------------
    # CALCULATE TRANSACTION SIGNALS
    # --------------------------------------------------------

    full_balance_drain = (
        oldbalance_org > 0
        and amount == oldbalance_org
        and newbalance_orig == 0
    )

    if oldbalance_org > 0:

        balance_drain_ratio = (
            amount / oldbalance_org
        )

    else:

        balance_drain_ratio = 0

    origin_balance_error = (
        oldbalance_org
        - amount
        - newbalance_orig
    )

    destination_balance_error = (
        oldbalance_dest
        + amount
        - newbalance_dest
    )

    destination_frozen = (
        oldbalance_dest == 0
        and newbalance_dest == 0
    )

    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    st.session_state.analysis_result = {

        "fraud_probability": fraud_probability,

        "risk_level": risk_level,

        "recommended_action": recommended_action,

        "transaction_type": transaction_type,

        "timestamp": transaction_timestamp,

        "full_balance_drain": full_balance_drain,

        "balance_drain_ratio": balance_drain_ratio,

        "origin_balance_error": origin_balance_error,

        "destination_balance_error": destination_balance_error,

        "destination_frozen": destination_frozen,

        "amount": amount,

        "oldbalance_org": oldbalance_org,

        "newbalance_orig": newbalance_orig,

        "oldbalance_dest": oldbalance_dest,

        "newbalance_dest": newbalance_dest

    }


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.analysis_result


if result is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">Decision Intelligence</div>',
        unsafe_allow_html=True
    )

    risk_level = result["risk_level"]

    if risk_level == "HIGH":

        risk_class = "risk-high"

        risk_description = (
            "The transaction contains patterns that require immediate investigation."
        )

    elif risk_level == "MEDIUM":

        risk_class = "risk-medium"

        risk_description = (
            "The transaction contains unusual behaviour that requires additional verification."
        )

    else:

        risk_class = "risk-low"

        risk_description = (
            "The transaction does not currently contain sufficient signals to classify it as high risk."
        )

    st.markdown(
        f"""
        <div class="risk-card {risk_class}">
            <div class="risk-title">{risk_level} RISK</div>
            <div class="risk-description">{risk_description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "Fraud Probability",
            f"{result['fraud_probability'] * 100:.2f}%"
        )

    with metric2:

        st.metric(
            "Risk Level",
            result["risk_level"]
        )

    with metric3:

        st.metric(
            "Recommended Action",
            result["recommended_action"]
        )

    with metric4:

        st.metric(
            "Transaction Type",
            result["transaction_type"]
        )

    st.caption(
        f"Analysis completed at {result['timestamp']}"
    )

    # ========================================================
    # EXPLANATION ENGINE
    # ========================================================

    st.markdown(
        '<div class="section-title">Model Reasoning</div>',
        unsafe_allow_html=True
    )

    explanations = []

    transaction_type = result["transaction_type"]

    # --------------------------------------------------------
    # BALANCE DRAIN
    # --------------------------------------------------------

    if result["full_balance_drain"]:

        explanations.append(
            "The transaction drained the sender's entire available balance. "
            "This is a high-risk financial pattern because the account balance "
            "reached zero immediately after the transaction."
        )

    # --------------------------------------------------------
    # BALANCE CONSISTENCY
    # --------------------------------------------------------

    if abs(result["origin_balance_error"]) > 0.01:

        explanations.append(
            "The sender balance relationship is inconsistent. "
            "The balance before the transaction minus the transaction amount "
            "does not match the reported balance after the transaction."
        )

    # --------------------------------------------------------
    # DRAIN RATIO
    # --------------------------------------------------------

    if result["balance_drain_ratio"] >= 0.9:

        explanations.append(
            "The transaction consumed an unusually large proportion "
            "of the sender's available balance."
        )

    # --------------------------------------------------------
    # DESTINATION PATTERN
    # --------------------------------------------------------

    if abs(result["destination_balance_error"]) > 0.01:

        explanations.append(
            "The destination balance relationship contains an unusual pattern "
            "because the expected balance movement does not match the reported balance."
        )

    # --------------------------------------------------------
    # FROZEN DESTINATION
    # --------------------------------------------------------

    if (
        transaction_type == "TRANSFER"
        and result["destination_frozen"]
    ):

        explanations.append(
            "The transfer was directed to a destination account whose recorded "
            "balance remained unchanged despite the transaction amount."
        )

    # --------------------------------------------------------
    # TRANSACTION TYPE
    # --------------------------------------------------------

    if transaction_type == "TRANSFER":

        explanations.append(
            "TRANSFER transactions receive additional scrutiny because the "
            "dataset analysis identified transfers as a major transaction pathway "
            "associated with fraudulent activity."
        )

    elif transaction_type == "CASH_OUT":

        explanations.append(
            "CASH_OUT transactions receive additional scrutiny because they "
            "represent the conversion of account value into cash and can form "
            "part of suspicious fund movement patterns."
        )

    elif transaction_type == "PAYMENT":

        explanations.append(
            "This is a PAYMENT transaction. The system evaluates the transaction "
            "using balance consistency, amount behaviour, and other engineered signals."
        )

    elif transaction_type == "CASH_IN":

        explanations.append(
            "This is a CASH_IN transaction. The system evaluates whether the "
            "balance movement is consistent with the reported transaction."
        )

    elif transaction_type == "DEBIT":

        explanations.append(
            "This is a DEBIT transaction. The system evaluates the transaction "
            "against balance movement and behavioural risk signals."
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if len(explanations) == 0:

        explanations.append(
            "No dominant mathematical anomaly was identified in the manually "
            "entered transaction values. The model's decision is based on the "
            "combined feature pattern."
        )

    # --------------------------------------------------------
    # DISPLAY EXPLANATIONS
    # --------------------------------------------------------

    for index, explanation in enumerate(
        explanations[:8],
        start=1
    ):

        st.markdown(
            f"""
            <div class="reason-card">
                <strong>{index}.</strong> {explanation}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # BALANCE CONSISTENCY CHECK
    # ========================================================

    st.markdown(
        '<div class="section-title">Transaction Integrity Checks</div>',
        unsafe_allow_html=True
    )

    integrity_col1, integrity_col2, integrity_col3 = st.columns(3)

    with integrity_col1:

        if abs(result["origin_balance_error"]) < 0.01:

            st.success(
                "Sender balance relationship: CONSISTENT"
            )

        else:

            st.error(
                "Sender balance relationship: INCONSISTENT"
            )

    with integrity_col2:

        if abs(result["destination_balance_error"]) < 0.01:

            st.success(
                "Recipient balance relationship: CONSISTENT"
            )

        else:

            st.warning(
                "Recipient balance relationship: UNUSUAL"
            )

    with integrity_col3:

        if result["full_balance_drain"]:

            st.warning(
                "Full balance drain detected"
            )

        else:

            st.success(
                "No full balance drain detected"
            )

    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "View technical model explanation"
    ):

        st.write(
            "The model evaluates engineered mathematical features derived "
            "from the transaction structure."
        )

        technical_data = pd.DataFrame(
            {
                "Signal": [
                    "Transaction Type",
                    "Transaction Amount",
                    "Balance Drain Ratio",
                    "Full Balance Drain",
                    "Origin Balance Error",
                    "Destination Balance Error",
                    "Destination Frozen"
                ],

                "Value": [
                    result["transaction_type"],
                    f"${result['amount']:,.2f}",
                    f"{result['balance_drain_ratio']:.4f}",
                    str(result["full_balance_drain"]),
                    f"{result['origin_balance_error']:,.2f}",
                    f"{result['destination_balance_error']:,.2f}",
                    str(result["destination_frozen"])
                ]
            }
        )

        st.dataframe(
            technical_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TEST SCENARIOS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Demonstration Scenarios</div>',
    unsafe_allow_html=True
)

st.caption(
    "These scenarios demonstrate how different transaction structures can create "
    "different balance and risk signals."
)

scenario_col1, scenario_col2, scenario_col3 = st.columns(3)


with scenario_col1:

    st.markdown(
        """
        <div class="scenario-card">
        <strong>Scenario 1 — TRANSFER</strong><br><br>
        Sender balance decreases, but the destination balance remains unchanged.
        This creates a destination balance inconsistency.
        </div>
        """,
        unsafe_allow_html=True
    )


with scenario_col2:

    st.markdown(
        """
        <div class="scenario-card">
        <strong>Scenario 2 — CASH_OUT</strong><br><br>
        The sender's balance is drained completely while the destination balance
        does not reflect the expected movement of funds.
        </div>
        """,
        unsafe_allow_html=True
    )


with scenario_col3:

    st.markdown(
        """
        <div class="scenario-card">
        <strong>Scenario 3 — PAYMENT</strong><br><br>
        The reported sender balance changes by an amount that does not match
        the payment amount, creating a balance integrity anomaly.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        ForensIQ AI | Explainable Transaction Risk Intelligence
    </div>
    """,
    unsafe_allow_html=True
)