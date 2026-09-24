"""
Generate a local SHAP explanation for one test-set accident.

The script:
    1. Loads the final XGBoost model.
    2. Loads the saved preprocessing pipeline.
    3. Recreates the fixed dataset split.
    4. Selects one test-set record.
    5. Predicts its accident-severity class.
    6. Calculates SHAP values for that record.
    7. Extracts contributions for the predicted class.
    8. Saves positive and negative feature contributions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import joblib
import pandas as pd
import shap

from src.models.training_pipeline import (
    prepare_training_data,
)

from src.explainability.shap_explainer import (
    calculate_shap_values,
    create_explainer,
    get_local_feature_contributions,
    get_top_negative_contributions,
    get_top_positive_contributions,
    save_local_contributions,
)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final"
    / "features.csv"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final"
    / "target.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "xgboost_final.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "xgboost_final_preprocessor.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "shap_local"
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SAMPLE_INDEX = 0

CLASS_NAMES = [
    "Fatal",
    "Serious",
    "Slight",
]


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("LOCAL SHAP EXPLANATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Verify artifacts
    # ---------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: "
            f"{PREPROCESSOR_PATH}"
        )

    # ---------------------------------------------------------------
    # Load model and preprocessor
    # ---------------------------------------------------------------

    print("\nLoading final model...")

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    # ---------------------------------------------------------------
    # Recreate fixed dataset split
    # ---------------------------------------------------------------

    print("Loading dataset...")

    (
        splits,
        _,
        _,
        _,
    ) = prepare_training_data(
        FEATURES_PATH,
        TARGET_PATH,
    )

    if SAMPLE_INDEX >= len(
        splits.X_test
    ):
        raise IndexError(
            "Sample index is outside the test set."
        )

    X_sample = splits.X_test.iloc[
        [SAMPLE_INDEX]
    ]

    y_actual = splits.y_test.iloc[
        SAMPLE_INDEX
    ]

    print(
        f"\nTest sample index: "
        f"{SAMPLE_INDEX}"
    )

    print(
        f"Actual class: "
        f"{y_actual}"
    )

    # ---------------------------------------------------------------
    # Transform sample
    # ---------------------------------------------------------------

    X_processed = (
        preprocessor.transform(
            X_sample
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    # ---------------------------------------------------------------
    # Model prediction
    # ---------------------------------------------------------------

    prediction = model.predict(
        X_processed
    )[0]

    probabilities = model.predict_proba(
        X_processed
    )[0]

    predicted_class = int(
        prediction
    )

    predicted_class_name = (
        CLASS_NAMES[predicted_class]
    )

    print(
        f"Predicted class: "
        f"{predicted_class_name}"
    )

    print("\nPrediction probabilities:")

    for class_index, probability in enumerate(
        probabilities
    ):
        print(
            f"  {CLASS_NAMES[class_index]:8s}: "
            f"{probability:.4f}"
        )

    # ---------------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------------

    print("\nCalculating SHAP values...")

    explainer = create_explainer(
        model
    )

    shap_values = calculate_shap_values(
        explainer,
        X_processed,
    )

    # ---------------------------------------------------------------
    # Extract predicted-class contributions
    # ---------------------------------------------------------------

    contributions = (
        get_local_feature_contributions(
            shap_values=shap_values,
            feature_names=feature_names,
            sample_index=0,
            class_index=predicted_class,
        )
    )

    positive = (
        get_top_positive_contributions(
            contributions,
            top_n=15,
        )
    )

    negative = (
        get_top_negative_contributions(
            contributions,
            top_n=15,
        )
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    all_path = (
        OUTPUT_DIR
        / "sample_0_all_contributions.csv"
    )

    positive_path = (
        OUTPUT_DIR
        / "sample_0_positive_contributions.csv"
    )

    negative_path = (
        OUTPUT_DIR
        / "sample_0_negative_contributions.csv"
    )

    save_local_contributions(
        contributions,
        all_path,
    )

    save_local_contributions(
        positive,
        positive_path,
    )

    save_local_contributions(
        negative,
        negative_path,
    )

    # ---------------------------------------------------------------
    # Display explanation
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"LOCAL EXPLANATION → "
        f"{predicted_class_name}"
    )
    print("=" * 70)

    print("\nTop features pushing TOWARD prediction:")
    print("-" * 70)

    if positive.empty:
        print("No positive contributions found.")
    else:
        print(
            positive[
                [
                    "feature",
                    "shap_value",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nTop features pushing AWAY from prediction:")
    print("-" * 70)

    if negative.empty:
        print("No negative contributions found.")
    else:
        print(
            negative[
                [
                    "feature",
                    "shap_value",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(all_path)
    print(positive_path)
    print(negative_path)

    print("\n" + "=" * 70)
    print("LOCAL SHAP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()