"""
Run LIME local explainability for the final XGBoost model.
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
import numpy as np
import pandas as pd

from src.models.training_pipeline import (
    prepare_training_data,
)

from src.explainability.lime_explainer import (
    create_explainer,
    explain_instance,
    explanation_to_dataframe,
    save_explanation,
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
    / "lime_local"
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SAMPLE_INDEX = 0
NUM_FEATURES = 15


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("LIME LOCAL EXPLAINABILITY")
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

    print("\nLoading final XGBoost model...")

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
            "Sample index is outside "
            "the test set."
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
    # Transform data
    # ---------------------------------------------------------------

    X_train_processed = (
        preprocessor.transform(
            splits.X_train
        )
    )

    X_sample_processed = (
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
    # Prediction
    # ---------------------------------------------------------------

    probabilities = model.predict_proba(
        X_sample_processed
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    class_names = [
        "Fatal",
        "Serious",
        "Slight",
    ]

    predicted_class = class_names[
        prediction
    ]

    print(
        f"Predicted class: "
        f"{predicted_class}"
    )

    print("\nPrediction probabilities:")

    for index, probability in enumerate(
        probabilities
    ):
        print(
            f"  {class_names[index]:8s}: "
            f"{probability:.4f}"
        )

    # ---------------------------------------------------------------
    # Prediction function for LIME
    # ---------------------------------------------------------------

    def predict_function(
        data: np.ndarray,
    ) -> np.ndarray:
        """
        LIME calls this function on perturbed samples.
        """

        return model.predict_proba(
            data
        )

    # ---------------------------------------------------------------
    # Create LIME explainer
    # ---------------------------------------------------------------

    print("\nCreating LIME explainer...")

    explainer = create_explainer(
        X_train=X_train_processed.toarray(),
        feature_names=feature_names,
    )

    # ---------------------------------------------------------------
    # Generate explanation
    # ---------------------------------------------------------------

    print("Generating local explanation...")

    instance = (
        X_sample_processed
        .toarray()[0]
    )

    explanation = explain_instance(
        explainer=explainer,
        instance=instance,
        predict_function=predict_function,
        num_features=NUM_FEATURES,
    )

    # ---------------------------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------------------------

    explanation_df = (
        explanation_to_dataframe(
            explanation
        )
    )

    output_path = (
        OUTPUT_DIR
        / "sample_0_explanation.csv"
    )

    save_explanation(
        explanation_df,
        output_path,
    )

    # ---------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"LIME EXPLANATION → "
        f"{predicted_class}"
    )
    print("=" * 70)

    print(
        explanation_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("LIME ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()