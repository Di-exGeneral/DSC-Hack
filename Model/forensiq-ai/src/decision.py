from pathlib import Path

import joblib
import pandas as pd
import shap

from src.features import create_features
from src.risk_engine import make_risk_decision


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "forensiq_model.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    artifact = joblib.load(
        MODEL_PATH
    )

    return (
        artifact["model"],
        artifact["feature_names"]
    )


# ============================================================
# PREPARE TRANSACTION
# ============================================================

def prepare_transaction(
    transaction,
    feature_names
):

    df = pd.DataFrame(
        [transaction]
    )

    # Create mathematical features

    df = create_features(
        df
    )

    # Remove columns that were not
    # used as model features

    df = df.drop(
        columns=[
            "isFraud",
            "isFlaggedFraud",
            "nameOrig",
            "nameDest"
        ],
        errors="ignore"
    )

    # Encode transaction type

    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )

    # Match training feature structure

    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    return df


# ============================================================
# GENERATE SHAP EXPLANATION
# ============================================================

def get_explanation(
    model,
    X
):

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    # Handle different SHAP versions

    if isinstance(
        shap_values,
        list
    ):

        values = shap_values[1][0]

    else:

        values = shap_values[0]

    explanation = pd.DataFrame({

        "feature": X.columns,

        "value": X.iloc[0].values,

        "shap_value": values

    })

    explanation["absolute_impact"] = (

        explanation["shap_value"]
        .abs()

    )

    explanation = explanation.sort_values(

        "absolute_impact",

        ascending=False

    )

    return explanation


# ============================================================
# CONVERT FEATURE INTO HUMAN EXPLANATION
# ============================================================

def explain_feature(
    feature,
    value,
    shap_value
):

    direction = (

        "increased"
        if shap_value > 0
        else
        "reduced"

    )

    explanations = {

        "full_balance_drain":
            "The transaction drained the sender's entire available balance.",

        "transfer_destination_frozen":
            "The transaction was a transfer where the destination balance remained at zero.",

        "balance_drain_ratio":
            "The transaction consumed an unusually large proportion of the sender's balance.",

        "origin_balance_error_abs":
            "The sender balance relationship contains an inconsistency.",

        "destination_balance_error_abs":
            "The destination balance relationship contains an unusual pattern.",

        "amount":
            "The transaction amount contributed to the model's decision.",

        "oldbalanceOrg":
            "The sender's original balance contributed to the risk assessment.",

        "newbalanceOrig":
            "The sender's resulting balance contributed to the risk assessment.",

        "is_risky_type":
            "The transaction belongs to a transaction type associated with elevated risk patterns.",

        "is_transfer":
            "The transaction is a transfer.",

        "is_cash_out":
            "The transaction is a cash-out."

    }

    explanation = explanations.get(

        feature,

        f"The feature '{feature}' influenced the model's decision."

    )

    return (

        explanation
        + f" This factor {direction} the fraud risk."

    )


# ============================================================
# COMPLETE FORENSIQ DECISION
# ============================================================

def analyze_transaction(
    transaction
):

    # Load trained model

    model, feature_names = load_model()

    # Prepare transaction

    X = prepare_transaction(

        transaction,

        feature_names

    )

    # Fraud probability

    fraud_probability = model.predict_proba(

        X

    )[0][1]

    # Risk decision

    decision = make_risk_decision(

        fraud_probability

    )

    # SHAP explanation

    explanation = get_explanation(

        model,

        X

    )

    # Keep only factors
    # that increased risk

    risk_factors = explanation[

        explanation["shap_value"] > 0

    ].head(5)

    human_reasons = []

    for _, row in risk_factors.iterrows():

        reason = explain_feature(

            row["feature"],

            row["value"],

            row["shap_value"]

        )

        human_reasons.append(

            reason

        )

    return {

        "decision": decision,

        "explanation": explanation,

        "human_reasons": human_reasons

    }


# ============================================================
# PRINT FORENSIQ REPORT
# ============================================================

def print_report(
    transaction,
    result
):

    decision = result["decision"]

    print("\n")

    print("=" * 60)

    print(
        "                 FORENSIQ AI"
    )

    print(
        "             TRANSACTION REPORT"
    )

    print("=" * 60)

    print()

    print(
        f"Transaction Type: "
        f"{transaction['type']}"
    )

    print(
        f"Transaction Amount: "
        f"{transaction['amount']:,.2f}"
    )

    print()

    print(
        f"Fraud Probability: "
        f"{decision['fraud_probability']:.2%}"
    )

    print(
        f"Risk Level: "
        f"{decision['risk_level']}"
    )

    print(
        f"Recommended Action: "
        f"{decision['recommended_action']}"
    )

    print()

    print("-" * 60)

    print(
        "WHY DID FORENSIQ MAKE THIS DECISION?"
    )

    print("-" * 60)

    for index, reason in enumerate(

        result["human_reasons"],

        start=1

    ):

        print(
            f"{index}. {reason}"
        )

    print()

    print("=" * 60)


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    transaction = {

        "step": 100,

        "type": "TRANSFER",

        "amount": 500000,

        "nameOrig": "C123456789",

        "oldbalanceOrg": 500000,

        "newbalanceOrig": 0,

        "nameDest": "C987654321",

        "oldbalanceDest": 0,

        "newbalanceDest": 0

    }

    result = analyze_transaction(

        transaction

    )

    print_report(

        transaction,

        result

    )