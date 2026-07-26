from pathlib import Path

import joblib
import pandas as pd

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
# PREDICT TRANSACTION
# ============================================================

def predict_transaction(transaction):

    model, feature_names = load_model()

    # Convert dictionary to DataFrame

    df = pd.DataFrame(
        [transaction]
    )

    # Create the same mathematical
    # features used during training

    df = create_features(df)

    # Remove identifiers
    # because they were not used by the model

    df = df.drop(
        columns=[
            "nameOrig",
            "nameDest",
            "isFraud",
            "isFlaggedFraud"
        ],
        errors="ignore"
    )

    # Convert transaction type

    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )

    # Make sure the prediction data
    # has exactly the same features
    # as the trained model

    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Generate fraud probability

    fraud_probability = model.predict_proba(
        df
    )[:, 1][0]

    # Apply risk engine

    decision = make_risk_decision(
        fraud_probability
    )

    return decision


# ============================================================
# TEST TRANSACTION
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

    result = predict_transaction(
        transaction
    )

    print(
        "\nFORensiQ TRANSACTION DECISION"
    )

    print(
        "=============================="
    )

    print(
        f"Fraud probability: "
        f"{result['fraud_probability']}"
    )

    print(
        f"Risk level: "
        f"{result['risk_level']}"
    )

    print(
        f"Recommended action: "
        f"{result['recommended_action']}"
    )