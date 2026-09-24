"""
Final XGBoost training and untouched test-set evaluation.

Workflow:
    1. Load engineered features and target.
    2. Recreate the fixed 60/20/20 split.
    3. Combine train + validation.
    4. Fit preprocessing only on train + validation.
    5. Train final XGBoost model.
    6. Evaluate once on the untouched test set.
    7. Save final model, preprocessor, and metrics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# -------------------------------------------------------------------
# Project import path
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.class_weights import create_sample_weights
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

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

FINAL_MODEL_PATH = (
    ARTIFACT_DIR
    / "xgboost_final.joblib"
)

FINAL_PREPROCESSOR_PATH = (
    ARTIFACT_DIR
    / "xgboost_final_preprocessor.joblib"
)

FINAL_METRICS_PATH = (
    METRICS_DIR
    / "xgboost_final_test.json"
)

CONFUSION_MATRIX_PATH = (
    METRICS_DIR
    / "xgboost_final_confusion_matrix.csv"
)

CLASSIFICATION_REPORT_PATH = (
    METRICS_DIR
    / "xgboost_final_classification_report.csv"
)


# -------------------------------------------------------------------
# XGBoost label mapping
# -------------------------------------------------------------------

LABEL_MAPPING = {
    "Fatal": 0,
    "Serious": 1,
    "Slight": 2,
}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("FINAL XGBOOST TRAINING")
    print("=" * 70)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Load and split data
    # ---------------------------------------------------------------

    (
        splits,
        preprocessor,
        numerical_columns,
        categorical_columns,
    ) = prepare_training_data(
        FEATURES_PATH,
        TARGET_PATH,
    )

    print("\nOriginal split:")
    print(f"Train      : {len(splits.X_train):,}")
    print(f"Validation : {len(splits.X_validation):,}")
    print(f"Test       : {len(splits.X_test):,}")

    # ---------------------------------------------------------------
    # Combine train + validation
    # ---------------------------------------------------------------

    X_train_final = pd.concat(
        [
            splits.X_train,
            splits.X_validation,
        ],
        axis=0,
    )

    y_train_final = pd.concat(
        [
            splits.y_train,
            splits.y_validation,
        ],
        axis=0,
    )

    print("\nFinal training data:")
    print(
        f"Train + Validation: "
        f"{len(X_train_final):,}"
    )

    print(
        f"Untouched Test    : "
        f"{len(splits.X_test):,}"
    )

    # ---------------------------------------------------------------
    # Fit preprocessing ONLY on train + validation
    # ---------------------------------------------------------------

    print("\nFitting preprocessing...")

    X_train_processed = (
        preprocessor.fit_transform(
            X_train_final
        )
    )

    X_test_processed = (
        preprocessor.transform(
            splits.X_test
        )
    )

    print("\nProcessed shapes:")
    print(
        f"Train + Validation: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Test              : "
        f"{X_test_processed.shape}"
    )

    # ---------------------------------------------------------------
    # Encode target
    # ---------------------------------------------------------------

    y_train_encoded = (
        y_train_final
        .map(LABEL_MAPPING)
        .astype(int)
        .to_numpy()
    )

    y_test_encoded = (
        splits.y_test
        .map(LABEL_MAPPING)
        .astype(int)
        .to_numpy()
    )

    # ---------------------------------------------------------------
    # Create balanced sample weights
    # ---------------------------------------------------------------

    sample_weights = (
        create_sample_weights(
            y_train_final
        ).to_numpy()
    )

    # ---------------------------------------------------------------
    # Create final XGBoost model
    # ---------------------------------------------------------------

    model = create_xgboost()

    print("\nTraining final XGBoost model...")

    start_time = time.perf_counter()

    model.fit(
        X_train_processed,
        y_train_encoded,
        sample_weight=sample_weights,
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds."
    )

    # ---------------------------------------------------------------
    # Untouched test evaluation
    # ---------------------------------------------------------------

    print("\nEvaluating on untouched test set...")

    y_pred = model.predict(
        X_test_processed
    )

    y_probability = model.predict_proba(
        X_test_processed
    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_test_encoded,
        y_pred,
    )

    macro_precision = precision_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test_encoded,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test_encoded,
        y_probability,
        multi_class="ovr",
        average="macro",
    )

    # ---------------------------------------------------------------
    # Print metrics
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST METRICS")
    print("=" * 70)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")
    print(f"ROC-AUC         : {roc_auc:.4f}")

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    matrix = confusion_matrix(
        y_test_encoded,
        y_pred,
        labels=[0, 1, 2],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Fatal",
            "Serious",
            "Slight",
        ],
        columns=[
            "Predicted_Fatal",
            "Predicted_Serious",
            "Predicted_Slight",
        ],
    )

    matrix_df.to_csv(
        CONFUSION_MATRIX_PATH
    )

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------

    report = classification_report(
        y_test_encoded,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "Fatal",
            "Serious",
            "Slight",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        CLASSIFICATION_REPORT_PATH
    )

    # ---------------------------------------------------------------
    # Save final metrics
    # ---------------------------------------------------------------

    metrics = {
        "model": "XGBoost",
        "evaluation_split": "untouched_test",
        "train_rows": int(
            len(X_train_final)
        ),
        "test_rows": int(
            len(splits.X_test)
        ),
        "training_time_seconds": float(
            training_time
        ),
        "accuracy": float(
            accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "roc_auc": float(
            roc_auc
        ),
        "label_mapping": LABEL_MAPPING,
        "feature_count": int(
            X_train_processed.shape[1]
        ),
    }

    with open(
        FINAL_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ---------------------------------------------------------------
    # Save final model and preprocessor
    # ---------------------------------------------------------------

    joblib.dump(
        model,
        FINAL_MODEL_PATH,
    )

    joblib.dump(
        preprocessor,
        FINAL_PREPROCESSOR_PATH,
    )

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------

    print("\nArtifacts saved:")
    print(
        f"Model        : {FINAL_MODEL_PATH}"
    )
    print(
        f"Preprocessor : {FINAL_PREPROCESSOR_PATH}"
    )
    print(
        f"Metrics      : {FINAL_METRICS_PATH}"
    )
    print(
        f"Confusion    : {CONFUSION_MATRIX_PATH}"
    )
    print(
        f"Report       : {CLASSIFICATION_REPORT_PATH}"
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()