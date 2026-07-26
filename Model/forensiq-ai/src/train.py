from pathlib import Path

import joblib
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.data import load_data
from src.features import create_features
from src.behavior import create_behavioral_features


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "forensiq_model.joblib"


# ============================================================
# CONFIGURATION
# ============================================================

# Use a sample while developing quickly.
#
# 1_000_000 = faster development
#
# Set to None when you are ready to train
# on the full 6.3 million-row dataset.

SAMPLE_SIZE = 1_000_000


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def prepare_data():

    print("Loading dataset...")

    df = load_data()

    print(
        f"Original dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # OPTIONAL FAST DEVELOPMENT SAMPLE
    # --------------------------------------------------------

    if SAMPLE_SIZE is not None:

        print(
            f"\nUsing a sample of "
            f"{SAMPLE_SIZE:,} rows "
            f"for faster development..."
        )

        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=42
        ).copy()

        print(
            f"Sampled dataset shape: {df.shape}"
        )

    # --------------------------------------------------------
    # CREATE MATHEMATICAL FRAUD FEATURES
    # --------------------------------------------------------

    print(
        "\nCreating mathematical features..."
    )

    df = create_features(df)

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    print(
        "Sorting transactions chronologically..."
    )

    df = df.sort_values(
        ["step", "nameOrig"]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # CREATE BEHAVIOURAL FEATURES
    # --------------------------------------------------------

    print(
        "Creating behavioural features..."
    )

    df = create_behavioral_features(df)

    print(
        "\nDataset after feature engineering:"
    )

    print(df.shape)

    return df


# ============================================================
# CHRONOLOGICAL DATA SPLIT
# ============================================================

def split_data(df):

    print(
        "\nCreating chronological data split..."
    )

    # 70% earliest transactions = training
    train_cutoff = df["step"].quantile(
        0.70
    )

    # 85% earliest transactions = validation boundary
    validation_cutoff = df["step"].quantile(
        0.85
    )

    train_df = df[
        df["step"] <= train_cutoff
    ].copy()

    validation_df = df[
        (df["step"] > train_cutoff)
        &
        (df["step"] <= validation_cutoff)
    ].copy()

    test_df = df[
        df["step"] > validation_cutoff
    ].copy()

    print(
        f"Training rows:   {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Testing rows:    {len(test_df):,}"
    )

    print(
        "\nFraud distribution:"
    )

    print(
        f"Training fraud:   "
        f"{train_df['isFraud'].sum():,}"
    )

    print(
        f"Validation fraud: "
        f"{validation_df['isFraud'].sum():,}"
    )

    print(
        f"Testing fraud:    "
        f"{test_df['isFraud'].sum():,}"
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# PREPARE MODEL FEATURES
# ============================================================

def prepare_features(
    train_df,
    validation_df,
    test_df
):

    print(
        "\nPreparing model features..."
    )

    # These columns must not be used as model features.

    drop_columns = [

        "isFraud",

        "isFlaggedFraud",

        "nameOrig",

        "nameDest"

    ]

    # --------------------------------------------------------
    # SEPARATE FEATURES FROM TARGET
    # --------------------------------------------------------

    X_train = train_df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    X_validation = validation_df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    X_test = test_df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # TARGET VARIABLE
    # --------------------------------------------------------

    y_train = train_df[
        "isFraud"
    ]

    y_validation = validation_df[
        "isFraud"
    ]

    y_test = test_df[
        "isFraud"
    ]

    # --------------------------------------------------------
    # ENCODE TRANSACTION TYPE
    # --------------------------------------------------------

    X_train = pd.get_dummies(
        X_train,
        columns=["type"],
        dtype=int
    )

    X_validation = pd.get_dummies(
        X_validation,
        columns=["type"],
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=["type"],
        dtype=int
    )

    # --------------------------------------------------------
    # ALIGN VALIDATION AND TEST COLUMNS
    # WITH TRAINING COLUMNS
    # --------------------------------------------------------

    X_validation = X_validation.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    print(
        f"Number of model features: "
        f"{X_train.shape[1]}"
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ============================================================
# TRAIN LIGHTGBM MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    X_validation,
    y_validation
):

    print(
        "\nTraining LightGBM model..."
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

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print(
        "\nGenerating predictions..."
    )

    # --------------------------------------------------------
    # FRAUD PROBABILITY
    # --------------------------------------------------------

    fraud_probability = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # DECISION THRESHOLD
    # --------------------------------------------------------

    threshold = 0.50

    predictions = (

        fraud_probability >= threshold

    ).astype(int)

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = average_precision_score(

        y_test,

        fraud_probability

    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = roc_auc_score(

        y_test,

        fraud_probability

    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MODEL PERFORMANCE"
    )

    print(
        "=" * 60
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(

        classification_report(

            y_test,

            predictions,

            zero_division=0

        )

    )

    print(
        "\nConfusion Matrix:"
    )

    print(

        confusion_matrix(

            y_test,

            predictions

        )

    )

    return fraud_probability


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_names
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    artifact = {

        "model": model,

        "feature_names": list(

            feature_names

        )

    }

    joblib.dump(

        artifact,

        MODEL_PATH

    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_PATH
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    df = prepare_data()

    # --------------------------------------------------------
    # 2. CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    (

        train_df,

        validation_df,

        test_df

    ) = split_data(df)

    # --------------------------------------------------------
    # 3. PREPARE FEATURES
    # --------------------------------------------------------

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

        test_df

    )

    # --------------------------------------------------------
    # 4. TRAIN MODEL
    # --------------------------------------------------------

    model = train_model(

        X_train,

        y_train,

        X_validation,

        y_validation

    )

    # --------------------------------------------------------
    # 5. EVALUATE MODEL
    # --------------------------------------------------------

    evaluate_model(

        model,

        X_test,

        y_test

    )

    # --------------------------------------------------------
    # 6. SAVE MODEL
    # --------------------------------------------------------

    save_model(

        model,

        X_train.columns

    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()