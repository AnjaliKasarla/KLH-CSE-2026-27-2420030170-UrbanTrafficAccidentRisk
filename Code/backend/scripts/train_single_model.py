"""
Train one selected accident-severity model.

Usage:
    python scripts/train_single_model.py logistic_regression
    python scripts/train_single_model.py decision_tree
    python scripts/train_single_model.py random_forest
    python scripts/train_single_model.py svm
    python scripts/train_single_model.py xgboost
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.class_weights import create_sample_weights
from src.models.decision_tree import create_decision_tree
from src.models.logistic_regression import create_logistic_regression
from src.models.random_forest import create_random_forest
from src.models.svm import create_svm
from src.models.trainer import evaluate_model, result_to_dict
from src.models.training_pipeline import prepare_training_data
from src.models.xgboost_model import create_xgboost


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


MODEL_FACTORIES = {
    "logistic_regression": create_logistic_regression,
    "decision_tree": create_decision_tree,
    "random_forest": create_random_forest,
    "svm": create_svm,
    "xgboost": create_xgboost,
}


# XGBoost requires integer labels.
XGB_LABEL_MAPPING = {
    "Fatal": 0,
    "Serious": 1,
    "Slight": 2,
}

XGB_INVERSE_LABEL_MAPPING = {
    0: "Fatal",
    1: "Serious",
    2: "Slight",
}


def main() -> None:
    """Train and evaluate one selected model."""

    if len(sys.argv) != 2:
        available = ", ".join(MODEL_FACTORIES.keys())

        raise SystemExit(
            f"Usage: python scripts/train_single_model.py "
            f"<model>\n\n"
            f"Available models: {available}"
        )

    model_name = sys.argv[1].lower()

    if model_name not in MODEL_FACTORIES:
        available = ", ".join(MODEL_FACTORIES.keys())

        raise SystemExit(
            f"Unknown model: {model_name}\n"
            f"Available models: {available}"
        )

    print("=" * 70)
    print(f"TRAINING MODEL: {model_name.upper()}")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------------

    print("\n[1/5] Preparing data...")

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

    # ---------------------------------------------------------------
    # 2. Fit preprocessing on training data only
    # ---------------------------------------------------------------

    print("\n[2/5] Fitting preprocessing pipeline...")

    X_train = preprocessor.fit_transform(
        splits.X_train
    )

    X_validation = preprocessor.transform(
        splits.X_validation
    )

    print(
        f"Encoded train shape      : {X_train.shape}"
    )

    print(
        f"Encoded validation shape : {X_validation.shape}"
    )

    # ---------------------------------------------------------------
    # 3. Prepare class imbalance handling
    # ---------------------------------------------------------------

    print(
        "\n[3/5] Configuring class imbalance handling..."
    )

    sample_weights = None

    if model_name == "xgboost":

        sample_weights = create_sample_weights(
            splits.y_train
        )

        print(
            "XGBoost sample weights prepared."
        )

    else:

        print(
            "Using model-level class_weight='balanced'."
        )

    # ---------------------------------------------------------------
    # 4. Train model
    # ---------------------------------------------------------------

    print("\n[4/5] Training model...")

    model = MODEL_FACTORIES[
        model_name
    ]()

    start_time = time.perf_counter()

    if model_name == "xgboost":

        # -----------------------------------------------------------
        # XGBoost requires integer class labels.
        # Original labels:
        # Fatal   -> 0
        # Serious -> 1
        # Slight  -> 2
        # -----------------------------------------------------------

        y_train_xgb = splits.y_train.map(
            XGB_LABEL_MAPPING
        )

        y_validation_xgb = splits.y_validation.map(
            XGB_LABEL_MAPPING
        )

        if y_train_xgb.isna().any():
            raise ValueError(
                "XGBoost label encoding failed for training labels."
            )

        if y_validation_xgb.isna().any():
            raise ValueError(
                "XGBoost label encoding failed for validation labels."
            )

        model.fit(
            X_train,
            y_train_xgb,
            sample_weight=sample_weights.to_numpy(),
            eval_set=[
                (
                    X_validation,
                    y_validation_xgb,
                )
            ],
            verbose=False,
        )

    else:

        # sklearn models already use
        # class_weight="balanced".
        #
        # Do NOT pass sample_weight here because that would
        # double-weight the minority classes.

        model.fit(
            X_train,
            splits.y_train,
        )

    training_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Training time: {training_time:.2f} seconds"
    )

    # ---------------------------------------------------------------
    # 5. Evaluate
    # ---------------------------------------------------------------

    print(
        "\n[5/5] Evaluating on validation set..."
    )

    if model_name == "xgboost":

        # XGBoost predicts integer labels.
        # Decode them back to the original target labels
        # before passing them to the common evaluator.

        y_pred_encoded = model.predict(
            X_validation
        )

        y_pred_decoded = [
            XGB_INVERSE_LABEL_MAPPING[int(label)]
            for label in y_pred_encoded
        ]

        # Use the common evaluator with a small wrapper
        # that returns the decoded predictions.
        result = evaluate_model(
            model=model,
            X_test=X_validation,
            y_test=splits.y_validation,
            model_name=model_name,
            label_mapping=XGB_INVERSE_LABEL_MAPPING,
        )

    else:

        result = evaluate_model(
            model=model,
            X_test=X_validation,
            y_test=splits.y_validation,
            model_name=model_name,
        )

    print("\n" + "-" * 70)
    print("VALIDATION RESULTS")
    print("-" * 70)

    print(
        f"Accuracy         : {result.accuracy:.4f}"
    )

    print(
        f"Macro Precision  : {result.macro_precision:.4f}"
    )

    print(
        f"Macro Recall     : {result.macro_recall:.4f}"
    )

    print(
        f"Macro F1         : {result.macro_f1:.4f}"
    )

    print(
        f"Weighted F1      : {result.weighted_f1:.4f}"
    )

    print(
        f"ROC-AUC          : "
        f"{result.roc_auc:.4f}"
        if result.roc_auc is not None
        else "ROC-AUC          : N/A"
    )

    print("\nConfusion Matrix:")

    print(
        result.confusion_matrix
    )

    print("\nClassification Report:")

    for label, values in (
        result.classification_report.items()
    ):

        if isinstance(values, dict):

            print(
                f"{label:12s} "
                f"precision={values.get('precision', 0):.4f} "
                f"recall={values.get('recall', 0):.4f} "
                f"f1={values.get('f1-score', 0):.4f}"
            )

    # ---------------------------------------------------------------
    # Save model artifact
    # ---------------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": result.model,
        "preprocessor": preprocessor,
        "model_name": model_name,
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
    }

    if model_name == "xgboost":
        artifact["label_mapping"] = XGB_LABEL_MAPPING
        artifact["inverse_label_mapping"] = XGB_INVERSE_LABEL_MAPPING

    model_path = (
        MODEL_DIR
        / f"{model_name}.joblib"
    )

    joblib.dump(
        artifact,
        model_path,
    )

    # ---------------------------------------------------------------
    # Save validation metrics
    # ---------------------------------------------------------------

    metrics = result_to_dict(
        result
    )

    metrics["training_time_seconds"] = (
        training_time
    )

    if model_name == "xgboost":
        metrics["label_mapping"] = XGB_LABEL_MAPPING

    metrics_path = (
        METRICS_DIR
        / f"{model_name}_validation.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nArtifacts saved:")

    print(
        f"Model   : {model_path}"
    )

    print(
        f"Metrics : {metrics_path}"
    )

    print("\n" + "=" * 70)

    print(
        f"{model_name.upper()} TRAINING COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()