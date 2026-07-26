from pathlib import Path

import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data import load_data
from src.features import create_features


# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# LOAD DATA
# ==================================================

def prepare_data():

    print("Loading dataset...")

    df = load_data()

    print(
        f"Original shape: {df.shape}"
    )

    df = create_features(df)

    print(
        f"After feature engineering: {df.shape}"
    )

    # Chronological ordering

    df = df.sort_values(
        "step"
    ).reset_index(drop=True)

    return df


# ==================================================
# SPLIT DATA
# ==================================================

def split_data(df):

    train_cutoff = df["step"].quantile(
        0.70
    )

    validation_cutoff = df["step"].quantile(
        0.85
    )

    train_df = df[
        df["step"] <= train_cutoff
    ]

    validation_df = df[
        (df["step"] > train_cutoff)
        &
        (df["step"] <= validation_cutoff)
    ]

    test_df = df[
        df["step"] > validation_cutoff
    ]

    return (
        train_df,
        validation_df,
        test_df
    )


# ==================================================
# PREPARE FEATURES
# ==================================================

def prepare_features(
    train_df,
    validation_df,
    test_df,
    feature_set
):

    # Always remove these

    excluded_columns = [
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest"
    ]

    # Select requested features

    X_train = train_df[
        feature_set
    ].copy()

    X_validation = validation_df[
        feature_set
    ].copy()

    X_test = test_df[
        feature_set
    ].copy()

    # Target

    y_train = train_df[
        "isFraud"
    ]

    y_validation = validation_df[
        "isFraud"
    ]

    y_test = test_df[
        "isFraud"
    ]

    # Encode transaction type

    if "type" in X_train.columns:

        X_train = pd.get_dummies(
            X_train,
            columns=["type"]
        )

        X_validation = pd.get_dummies(
            X_validation,
            columns=["type"]
        )

        X_test = pd.get_dummies(
            X_test,
            columns=["type"]
        )

    # Align columns

    X_validation = X_validation.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ==================================================
# TRAIN AND EVALUATE ONE MODEL
# ==================================================

def run_experiment(
    model_name,
    feature_set,
    train_df,
    validation_df,
    test_df
):

    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = prepare_features(
        train_df,
        validation_df,
        test_df,
        feature_set
    )

    print(
        f"Features used: {X_train.shape[1]}"
    )

    model = LGBMClassifier(

        objective="binary",

        n_estimators=500,

        learning_rate=0.05,

        num_leaves=31,

        class_weight="balanced",

        random_state=42,

        n_jobs=-1,

        verbosity=-1

    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (
                X_validation,
                y_validation
            )
        ]

    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    results = {

        "Model": model_name,

        "Features": X_train.shape[1],

        "PR-AUC": average_precision_score(
            y_test,
            probabilities
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0
        )

    }

    print("\nRESULTS")

    for key, value in results.items():

        print(
            f"{key}: {value}"
        )

    return results


# ==================================================
# MAIN EXPERIMENT
# ==================================================

def main():

    df = prepare_data()

    (
        train_df,
        validation_df,
        test_df
    ) = split_data(df)

    # ----------------------------------------------
    # MODEL A
    # ALL FEATURES
    # ----------------------------------------------

    all_features = [

        "step",

        "type",

        "amount",

        "oldbalanceOrg",

        "newbalanceOrig",

        "oldbalanceDest",

        "newbalanceDest",

        "is_transfer",

        "is_cash_out",

        "is_risky_type",

        "full_balance_drain",

        "balance_drain_ratio",

        "origin_balance_error",

        "origin_balance_error_abs",

        "destination_balance_error",

        "destination_balance_error_abs",

        "destination_frozen",

        "transfer_destination_frozen",

        "risky_full_balance_drain",

        "log_amount"

    ]

    # ----------------------------------------------
    # MODEL B
    # REMOVE STEP
    # ----------------------------------------------

    without_step = [

        feature

        for feature in all_features

        if feature != "step"

    ]

    # ----------------------------------------------
    # MODEL C
    # RAW FEATURES ONLY
    # ----------------------------------------------

    raw_features = [

        "step",

        "type",

        "amount",

        "oldbalanceOrg",

        "newbalanceOrig",

        "oldbalanceDest",

        "newbalanceDest"

    ]

    # ----------------------------------------------
    # MODEL D
    # RAW + MATHEMATICAL FEATURES
    # ----------------------------------------------

    mathematical_features = [

        "step",

        "type",

        "amount",

        "oldbalanceOrg",

        "newbalanceOrig",

        "oldbalanceDest",

        "newbalanceDest",

        "full_balance_drain",

        "balance_drain_ratio",

        "origin_balance_error",

        "origin_balance_error_abs",

        "destination_balance_error",

        "destination_balance_error_abs",

        "destination_frozen",

        "transfer_destination_frozen",

        "risky_full_balance_drain",

        "log_amount"

    ]

    results = []

    results.append(

        run_experiment(

            "MODEL A - ALL FEATURES",

            all_features,

            train_df,

            validation_df,

            test_df

        )

    )

    results.append(

        run_experiment(

            "MODEL B - WITHOUT STEP",

            without_step,

            train_df,

            validation_df,

            test_df

        )

    )

    results.append(

        run_experiment(

            "MODEL C - RAW FEATURES ONLY",

            raw_features,

            train_df,

            validation_df,

            test_df

        )

    )

    results.append(

        run_experiment(

            "MODEL D - MATHEMATICAL FEATURES",

            mathematical_features,

            train_df,

            validation_df,

            test_df

        )

    )

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print("FINAL ABLATION COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    results_df.to_csv(
        PROJECT_ROOT
        / "reports"
        / "ablation_results.csv",

        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        "reports/ablation_results.csv"
    )


if __name__ == "__main__":

    main()