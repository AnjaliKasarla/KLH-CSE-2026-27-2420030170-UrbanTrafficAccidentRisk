"""
Run SHAP explainability for the final XGBoost model.
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

from src.models.training_pipeline import (
    prepare_training_data,
)

from src.explainability.shap_explainer import (
    calculate_global_importance,
    create_explainer,
    calculate_shap_values,
    save_global_importance,
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "shap_global_importance.csv"
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# SHAP is computationally expensive.
# We use a reproducible subset for explainability.
SHAP_SAMPLE_SIZE = 5000
RANDOM_STATE = 42


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("SHAP EXPLAINABILITY")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Verify artifacts
    # ---------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Final preprocessor not found: "
            f"{PREPROCESSOR_PATH}"
        )

    # ---------------------------------------------------------------
    # Load final model
    # ---------------------------------------------------------------

    print("\nLoading final XGBoost model...")

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    # ---------------------------------------------------------------
    # Load engineered data
    # ---------------------------------------------------------------

    print("Loading engineered dataset...")

    (
        splits,
        _,
        numerical_columns,
        categorical_columns,
    ) = prepare_training_data(
        FEATURES_PATH,
        TARGET_PATH,
    )

    # ---------------------------------------------------------------
    # Use untouched test set for explanation
    # ---------------------------------------------------------------

    X_test = splits.X_test.copy()

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_test),
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nSHAP samples: {len(X_sample):,}"
    )

    # ---------------------------------------------------------------
    # Transform using saved preprocessor
    # ---------------------------------------------------------------

    X_processed = (
        preprocessor.transform(
            X_sample
        )
    )

    print(
        f"Processed shape: "
        f"{X_processed.shape}"
    )

    # ---------------------------------------------------------------
    # Get feature names
    # ---------------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    print(
        f"Encoded features: "
        f"{len(feature_names)}"
    )

    # ---------------------------------------------------------------
    # Create SHAP explainer
    # ---------------------------------------------------------------

    print("\nCreating SHAP TreeExplainer...")

    explainer = create_explainer(
        model
    )

    # ---------------------------------------------------------------
    # Calculate SHAP values
    # ---------------------------------------------------------------

    print("Calculating SHAP values...")

    shap_values = calculate_shap_values(
        explainer,
        X_processed,
    )

    print("SHAP calculation completed.")

    # ---------------------------------------------------------------
    # Global feature importance
    # ---------------------------------------------------------------

    importance = calculate_global_importance(
        shap_values,
        feature_names,
    )

    save_global_importance(
        importance,
        OUTPUT_PATH,
    )

    # ---------------------------------------------------------------
    # Display top features
    # ---------------------------------------------------------------

    print("\nTop 20 SHAP Features")
    print("-" * 70)

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()