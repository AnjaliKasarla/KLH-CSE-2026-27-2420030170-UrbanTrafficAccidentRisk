"""
LIME explainability utilities for the final XGBoost accident-risk model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


CLASS_NAMES = [
    "Fatal",
    "Serious",
    "Slight",
]


def create_explainer(
    X_train: Any,
    feature_names: list[str],
) -> LimeTabularExplainer:
    """
    Create a LIME tabular explainer.

    The explainer uses the transformed numerical feature
    representation produced by the saved preprocessing pipeline.
    """

    return LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def explain_instance(
    explainer: LimeTabularExplainer,
    instance: Any,
    predict_function: Callable,
    num_features: int = 15,
):
    """
    Generate a local LIME explanation for one instance.
    """

    return explainer.explain_instance(
        instance,
        predict_function,
        num_features=num_features,
    )


def explanation_to_dataframe(
    explanation: Any,
) -> pd.DataFrame:
    """
    Convert a LIME explanation into a DataFrame.

    The resulting table contains:
        feature
        weight
    """

    rows = explanation.as_list()

    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "weight",
        ],
    )


def save_explanation(
    explanation_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save LIME explanation to CSV."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    explanation_df.to_csv(
        output_path,
        index=False,
    )