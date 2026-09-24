"""
Common model training utilities for the Urban Traffic Accident project.

Provides a reusable interface for training and evaluating
classification models on the prepared accident-risk dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelResult:
    """Container for model evaluation results."""

    model_name: str
    model: Any
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float | None
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]


def train_model(
    model: Any,
    X_train: Any,
    y_train: Any,
) -> Any:
    """
    Train a classification model.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Any,
    model_name: str,
    label_mapping: Dict[int, str] | None = None,
) -> ModelResult:
    """
    Evaluate a trained classification model.

    Metrics include:
        Accuracy
        Macro Precision
        Macro Recall
        Macro F1
        Weighted F1
        Multiclass ROC-AUC when probabilities are available
        Confusion Matrix
        Classification Report

    label_mapping:
        Optional mapping used when a model such as XGBoost
        internally represents class labels as integers.
    """

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------------
    # Decode model predictions when required
    # ---------------------------------------------------------------

    if label_mapping is not None:

        predictions = np.asarray(
            predictions
        ).astype(int)

        predictions = np.array(
            [
                label_mapping[int(label)]
                for label in predictions
            ]
        )

    # ---------------------------------------------------------------
    # Basic classification metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # ROC-AUC
    # ---------------------------------------------------------------

    roc_auc = None

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = model.predict_proba(
            X_test
        )

        try:

            # XGBoost returns probabilities in its internal
            # class order. For our fixed mapping this is:
            #
            # 0 -> Fatal
            # 1 -> Serious
            # 2 -> Slight

            if label_mapping is not None:

                encoded_y_test = np.asarray(
                    [
                        {
                            "Fatal": 0,
                            "Serious": 1,
                            "Slight": 2,
                        }[str(label)]
                        for label in y_test
                    ]
                )

                roc_auc = roc_auc_score(
                    encoded_y_test,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )

            else:

                roc_auc = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )

        except ValueError:

            roc_auc = None

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return ModelResult(
        model_name=model_name,
        model=model,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        roc_auc=roc_auc,
        confusion_matrix=matrix,
        classification_report=report,
    )


def result_to_dict(
    result: ModelResult,
) -> Dict[str, Any]:
    """
    Convert model results into a JSON-serializable dictionary.
    """

    return {
        "model_name": result.model_name,
        "accuracy": result.accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "weighted_f1": result.weighted_f1,
        "roc_auc": result.roc_auc,
        "confusion_matrix": result.confusion_matrix.tolist(),
        "classification_report": result.classification_report,
    }