"""
SHAP explainability utilities for the final XGBoost accident-risk model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap


def create_explainer(model: Any) -> shap.TreeExplainer:
    """
    Create a TreeSHAP explainer for an XGBoost tree model.
    """

    return shap.TreeExplainer(model)


def calculate_shap_values(
    explainer: shap.TreeExplainer,
    X: Any,
) -> Any:
    """
    Calculate SHAP values for the supplied feature matrix.
    """

    return explainer.shap_values(X)


def calculate_global_importance(
    shap_values: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP importance for each feature.

    For multiclass models, importance is aggregated across
    all output classes.
    """

    if isinstance(shap_values, list):

        importance = np.mean(
            [
                np.abs(values).mean(axis=0)
                for values in shap_values
            ],
            axis=0,
        )

    else:

        values = np.asarray(shap_values)

        if values.ndim == 3:
            importance = np.abs(values).mean(
                axis=(0, 2)
            )
        else:
            importance = np.abs(values).mean(
                axis=0
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


def save_global_importance(
    importance: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save global SHAP feature importance.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance.to_csv(
        output_path,
        index=False,
    )