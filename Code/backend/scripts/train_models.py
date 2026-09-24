"""
Train and evaluate baseline models for Urban Traffic Accident Severity.

Models:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - SVM
    - XGBoost

The preprocessing transformer is fitted only on the training split.
Class weights/sample weights are calculated only from training labels.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

from src.models.class_weights import create_sample_weights
from src.models.decision_tree import create_decision_tree
from src.models.logistic_regression import create_logistic_regression
from src.models.random_forest import create_random_forest
from src.models.svm import create_svm
from src.models.trainer import evaluate_model, result_to_dict
from src.models.training_pipeline import prepare_training_data
from src.models.xgboost_model import create_xgboost


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

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

# -------------------------------------------------------------------
# Model registry
# -------------------------------------------------------------------

MODELS = {
    "logistic_regression": create_logistic_regression,
    "decision_tree": create_decision_tree,
    "random_forest": create_random_forest,
    "svm": create_svm,
    "xgboost": create_xgboost,
}


# -------------------------------------------------------------------
# Training helpers
# -------------------------------------------------------------------

def train_single_model(
    model_name: str,
    model,
    preprocessor,
    X_train,
    y_train,
    X_validation,
    y_validation,
    sample_weights,
):
    """
    Fit preprocessing and model on training data, then evaluate
    on the validation set.
    """

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 70)

    start_time = time.perf_counter()

    # ---------------------------------------------------------------
    # Preprocessing is fitted ONLY on training data.
    # ---------------------------------------------------------------

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_validation_transformed = preprocessor.transform(
        X_validation
    )

    print(
        "Encoded training shape   :",
        X_train_transformed.shape,
    )

    print(
        "Encoded validation shape :",
        X_validation_transformed.shape,
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    if model_name == "xgboost":

        model.fit(
            X_train_transformed,
            y_train,
            sample_weight=sample_weights.to_numpy(),
            eval_set=[
                (
                    X_validation_transformed,
                    y_validation,
                )
            ],
            verbose=False,
        )

    else:

        model.fit(
            X_train_transformed,
            y_train,
            sample_weight=sample_weights.to_numpy(),
        )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    result = evaluate_model(
        model=model,
        X_test=X_validation_transformed,
        y_test=y_validation,
        model_name=model_name,
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"Training time           : {elapsed:.2f} seconds"
    )

    print(
        f"Accuracy                : {result.accuracy:.4f}"
    )

    print(
        f"Macro Precision        : {result.macro_precision:.4f}"
    )

    print(
        f"Macro Recall           : {result.macro_recall:.4f}"
    )

    print(
        f"Macro F1               : {result.macro_f1:.4f}"
    )

    print(
        f"Weighted F1            : {result.weighted_f1:.4f}"
    )

    print(
        f"ROC-AUC                : "
        f"{result.roc_auc if result.roc_auc is not None else 'N/A'}"
    )

    return (
        result,
        preprocessor,
        elapsed,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("URBAN TRAFFIC ACCIDENT - MODEL TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load and split
    # ---------------------------------------------------------------

    print("\n[1/4] Preparing training data...")

    (
        splits,
        preprocessor,
        numerical_columns,
        categorical_columns,
    ) = prepare_training_data(
        FEATURES_PATH,
        TARGET_PATH,
    )

    print(
        f"Training rows   : {len(splits.X_train):,}"
    )

    print(
        f"Validation rows : {len(splits.X_validation):,}"
    )

    print(
        f"Test rows       : {len(splits.X_test):,}"
    )

    print(
        f"Numerical       : {len(numerical_columns)}"
    )

    print(
        f"Categorical     : {len(categorical_columns)}"
    )

    # ---------------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------------

    print("\n[2/4] Calculating training class weights...")

    sample_weights = create_sample_weights(
        splits.y_train
    )

    class_weight_values = (
        sample_weights.groupby(
            splits.y_train
        ).first().to_dict()
    )

    print(
        "Class weights:",
        class_weight_values,
    )

    # ---------------------------------------------------------------
    # Prepare directories
    # ---------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Train models
    # ---------------------------------------------------------------

    print("\n[3/4] Training models...")

    results = {}

    for model_name, model_factory in MODELS.items():

        model = model_factory()

        try:

            result, fitted_preprocessor, elapsed = (
                train_single_model(
                    model_name=model_name,
                    model=model,
                    preprocessor=preprocessor,
                    X_train=splits.X_train,
                    y_train=splits.y_train,
                    X_validation=splits.X_validation,
                    y_validation=splits.y_validation,
                    sample_weights=sample_weights,
                )
            )

            results[model_name] = result_to_dict(
                result
            )

            artifact = {
                "model": result.model,
                "preprocessor": fitted_preprocessor,
                "model_name": model_name,
                "numerical_features": numerical_columns,
                "categorical_features": categorical_columns,
            }

            artifact_path = (
                MODEL_DIR
                / f"{model_name}.joblib"
            )

            joblib.dump(
                artifact,
                artifact_path,
            )

            print(
                f"Artifact saved: {artifact_path}"
            )

        except Exception as exc:

            print(
                f"\n{model_name} FAILED:"
            )

            print(
                repr(exc)
            )

    # ---------------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------------

    print("\n[4/4] Saving evaluation results...")

    metrics_path = (
        METRICS_DIR
        / "baseline_model_results.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nMetrics saved: {metrics_path}"
    )

    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING COMPLETED")
    print("=" * 70)

    for model_name, metrics in results.items():

        print(
            f"{model_name:22s} "
            f"Macro F1 = "
            f"{metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()