import numpy as np
import pandas as pd


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()

    print("Creating fast behavioral features...")

    # Sort chronologically
    data = data.sort_values(
        ["step", "nameOrig"]
    ).reset_index(drop=True)

    # ==================================================
    # SENDER HISTORY
    # ==================================================

    sender = data.groupby(
        "nameOrig",
        sort=False
    )

    # Previous transaction count
    data["sender_txn_count_before"] = (
        sender.cumcount()
    )

    # Previous amount
    data["sender_previous_amount"] = (
        sender["amount"]
        .shift(1)
        .fillna(0)
    )

    # Cumulative historical amount
    cumulative_amount = (
        sender["amount"]
        .cumsum()
    )

    # Previous cumulative amount
    previous_cumulative_amount = (
        cumulative_amount
        - data["amount"]
    )

    data["sender_avg_amount_before"] = np.where(

        data["sender_txn_count_before"] > 0,

        previous_cumulative_amount
        / data["sender_txn_count_before"],

        0
    )

    # Historical maximum using expanding maximum
    # This is still useful but faster than expanding mean/std
    data["sender_max_amount_before"] = (

        sender["amount"]
        .cummax()
        .groupby(data["nameOrig"])
        .shift(1)
        .fillna(0)
    )

    # ==================================================
    # AMOUNT ABNORMALITY
    # ==================================================

    data["amount_vs_sender_average"] = np.where(

        data["sender_avg_amount_before"] > 0,

        data["amount"]
        / data["sender_avg_amount_before"],

        0
    )

    data["amount_vs_sender_max"] = np.where(

        data["sender_max_amount_before"] > 0,

        data["amount"]
        / data["sender_max_amount_before"],

        0
    )

    # ==================================================
    # TIME SINCE PREVIOUS TRANSACTION
    # ==================================================

    previous_step = (
        sender["step"]
        .shift(1)
    )

    data["time_since_sender_previous_txn"] = (

        data["step"]
        - previous_step

    ).fillna(-1)

    # ==================================================
    # NEW ACCOUNT FLAGS
    # ==================================================

    data["new_sender"] = (

        data["sender_txn_count_before"] == 0

    ).astype(int)

    # ==================================================
    # DESTINATION HISTORY
    # ==================================================

    destination = data.groupby(
        "nameDest",
        sort=False
    )

    data["destination_txn_count_before"] = (

        destination.cumcount()
    )

    data["destination_previous_amount"] = (

        destination["amount"]
        .shift(1)
        .fillna(0)
    )

    # ==================================================
    # BEHAVIOURAL FLAGS
    # ==================================================

    data["unusually_large_for_sender"] = (

        (
            data["amount_vs_sender_average"] >= 3
        )

        &

        (
            data["sender_txn_count_before"] >= 2
        )

    ).astype(int)

    data["new_sender_large_transaction"] = (

        (
            data["new_sender"] == 1
        )

        &

        (
            data["amount"] > 100000
        )

    ).astype(int)

    print(
        "Behavioral features created successfully."
    )

    return data