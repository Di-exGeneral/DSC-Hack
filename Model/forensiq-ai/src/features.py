import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fraud-detection features from raw transaction data.

    These features are based on patterns discovered during EDA.
    """

    data = df.copy()

    # ==================================================
    # 1. TRANSACTION TYPE FEATURES
    # ==================================================

    data["is_transfer"] = (
        data["type"] == "TRANSFER"
    ).astype(int)

    data["is_cash_out"] = (
        data["type"] == "CASH_OUT"
    ).astype(int)

    data["is_risky_type"] = (
        data["type"].isin(
            ["TRANSFER", "CASH_OUT"]
        )
    ).astype(int)

    # ==================================================
    # 2. FULL BALANCE DRAIN
    # ==================================================

    data["full_balance_drain"] = (
        (data["amount"] == data["oldbalanceOrg"]) &
        (data["newbalanceOrig"] == 0)
    ).astype(int)

    # ==================================================
    # 3. BALANCE DRAIN RATIO
    # ==================================================

    data["balance_drain_ratio"] = np.where(
        data["oldbalanceOrg"] > 0,
        data["amount"] / data["oldbalanceOrg"],
        0
    )

    data["balance_drain_ratio"] = (
        data["balance_drain_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ==================================================
    # 4. SENDER BALANCE CONSISTENCY
    # ==================================================

    data["origin_balance_error"] = (
        data["oldbalanceOrg"]
        - data["amount"]
        - data["newbalanceOrig"]
    )

    data["origin_balance_error_abs"] = (
        data["origin_balance_error"].abs()
    )

    # ==================================================
    # 5. DESTINATION BALANCE CONSISTENCY
    # ==================================================

    data["destination_balance_error"] = (
        data["oldbalanceDest"]
        + data["amount"]
        - data["newbalanceDest"]
    )

    data["destination_balance_error_abs"] = (
        data["destination_balance_error"].abs()
    )

    # ==================================================
    # 6. DESTINATION BALANCE PATTERN
    # ==================================================

    data["destination_frozen"] = (
        (data["oldbalanceDest"] == 0) &
        (data["newbalanceDest"] == 0)
    ).astype(int)

    data["transfer_destination_frozen"] = (
        (data["type"] == "TRANSFER") &
        (data["oldbalanceDest"] == 0) &
        (data["newbalanceDest"] == 0)
    ).astype(int)

    # ==================================================
    # 7. INTERACTION FEATURES
    # ==================================================

    data["risky_full_balance_drain"] = (
        (data["is_risky_type"] == 1) &
        (data["full_balance_drain"] == 1)
    ).astype(int)

    # ==================================================
    # 8. LOG TRANSFORMATION
    # ==================================================

    data["log_amount"] = np.log1p(
        data["amount"]
    )

    return data