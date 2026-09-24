"""
SHAP explainability utilities for the final XGBoost accident-risk model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap


CLASS_NAMES = [
    "Fatal",
    "Serious",
    "Slight",
]


def create_explainer(model: Any) -> shap.TreeExplainer:
    """Create a TreeSHAP explainer for an XGBoost model."""

    return shap.TreeExplainer(model)


def calculate_shap_values(
    explainer: shap.TreeExplainer,
    X: Any,
) -> np.ndarray:
    """Calculate SHAP values."""

    return np.asarray(
        explainer.shap_values(X)
    )


def calculate_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP importance
    aggregated across all output classes.
    """

    values = np.asarray(shap_values)

    if values.ndim == 3:
        importance = np.abs(values).mean(
            axis=(0, 2)
        )

    elif values.ndim == 2:
        importance = np.abs(values).mean(
            axis=0
        )

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    if len(feature_names) != len(importance):
        raise ValueError(
            "Feature-name count does not match "
            "SHAP feature count."
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": importance,
        }
    )

    return result.sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(drop=True)


def get_class_shap_values(
    shap_values: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """
    Extract SHAP values for one output class.

    Expected input:
        (samples, features, classes)
    """

    values = np.asarray(shap_values)

    if values.ndim != 3:
        raise ValueError(
            "Expected multiclass SHAP values "
            "with shape "
            "(samples, features, classes)."
        )

    if not 0 <= class_index < values.shape[2]:
        raise ValueError(
            f"Invalid class index: {class_index}"
        )

    return values[:, :, class_index]


def get_local_feature_contributions(
    shap_values: np.ndarray,
    feature_names: list[str],
    sample_index: int,
    class_index: int,
) -> pd.DataFrame:
    """
    Return feature contributions for one sample
    and one predicted output class.
    """

    class_values = get_class_shap_values(
        shap_values,
        class_index,
    )

    if not 0 <= sample_index < class_values.shape[0]:
        raise ValueError(
            f"Invalid sample index: {sample_index}"
        )

    contributions = class_values[
        sample_index
    ]

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": contributions,
            "absolute_shap": np.abs(
                contributions
            ),
        }
    )

    return result.sort_values(
        "absolute_shap",
        ascending=False,
    ).reset_index(drop=True)


def get_top_positive_contributions(
    contributions: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return features pushing toward the selected class."""

    return (
        contributions[
            contributions["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def get_top_negative_contributions(
    contributions: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return features pushing away from the selected class."""

    return (
        contributions[
            contributions["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def save_global_importance(
    importance: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save global SHAP feature importance."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        output_path,
        index=False,
    )


def save_local_contributions(
    contributions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save local SHAP contributions."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    contributions.to_csv(
        output_path,
        index=False,
    )