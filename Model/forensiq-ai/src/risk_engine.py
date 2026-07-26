import pandas as pd


# ============================================================
# RISK THRESHOLDS
# ============================================================

LOW_RISK_THRESHOLD = 0.30

HIGH_RISK_THRESHOLD = 0.70


# ============================================================
# CONVERT PROBABILITY TO RISK LEVEL
# ============================================================

def get_risk_level(fraud_probability):

    if fraud_probability < LOW_RISK_THRESHOLD:

        return "LOW"

    elif fraud_probability < HIGH_RISK_THRESHOLD:

        return "MEDIUM"

    else:

        return "HIGH"


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def get_recommended_action(risk_level):

    if risk_level == "LOW":

        return "ALLOW"

    elif risk_level == "MEDIUM":

        return "STEP_UP_VERIFICATION"

    else:

        return "HOLD_AND_INVESTIGATE"


# ============================================================
# COMPLETE RISK DECISION
# ============================================================

def make_risk_decision(fraud_probability):

    risk_level = get_risk_level(
        fraud_probability
    )

    action = get_recommended_action(
        risk_level
    )

    return {

        "fraud_probability": round(
            float(fraud_probability),
            4
        ),

        "risk_level": risk_level,

        "recommended_action": action

    }


# ============================================================
# APPLY RISK ENGINE TO DATAFRAME
# ============================================================

def apply_risk_engine(
    df,
    fraud_probabilities
):

    result = df.copy()

    result["fraud_probability"] = (
        fraud_probabilities
    )

    result["risk_level"] = (

        result["fraud_probability"]
        .apply(get_risk_level)

    )

    result["recommended_action"] = (

        result["risk_level"]
        .apply(get_recommended_action)

    )

    return result