from pathlib import Path

import joblib
import pandas as pd
import shap

from src.features import create_features


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

    # Remove columns not used
    # by the model

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

    # Match model training columns

    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    return df


# ============================================================
# EXPLAIN TRANSACTION
# ============================================================

def explain_transaction(
    transaction
):

    model, feature_names = load_model()

    X = prepare_transaction(
        transaction,
        feature_names
    )

    # Fraud probability

    fraud_probability = model.predict_proba(
        X
    )[0][1]

    # SHAP explainer

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    # LightGBM binary classification
    # can return either a list or array
    # depending on SHAP version.

    if isinstance(
        shap_values,
        list
    ):

        values = shap_values[1][0]

    else:

        values = shap_values[0]

    # Create explanation table

    explanation = pd.DataFrame({

        "feature": X.columns,

        "value": X.iloc[0].values,

        "shap_value": values

    })

    # Sort by absolute impact

    explanation["absolute_impact"] = (

        explanation["shap_value"]
        .abs()

    )

    explanation = explanation.sort_values(

        "absolute_impact",

        ascending=False

    )

    return (
        fraud_probability,
        explanation
    )


# ============================================================
# TEST
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

    probability, explanation = (

        explain_transaction(
            transaction
        )

    )

    print(
        "\nFORensiQ EXPLANATION"
    )

    print(
        "===================="
    )

    print(

        f"\nFraud probability: "
        f"{probability:.4f}"

    )

    print(
        "\nTop factors influencing "
        "the decision:"
    )

    print(

        explanation[
            [
                "feature",
                "value",
                "shap_value"
            ]
        ].head(10).to_string(
            index=False
        )

    )