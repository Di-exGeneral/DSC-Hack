from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "forensiq_model.joblib"
)


def main():

    print("Loading trained model...")

    artifact = joblib.load(
        MODEL_PATH
    )

    model = artifact["model"]

    feature_names = artifact[
        "feature_names"
    ]

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({

        "feature": feature_names,

        "importance": importance

    })

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print("\nFEATURE IMPORTANCE")
    print("=" * 50)

    print(
        feature_importance
        .to_string(index=False)
    )


if __name__ == "__main__":

    main()