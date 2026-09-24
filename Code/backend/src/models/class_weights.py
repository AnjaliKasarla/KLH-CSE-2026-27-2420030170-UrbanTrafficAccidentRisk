"""
Class-weight utilities for imbalanced accident-severity classification.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(
    y_train: pd.Series,
) -> Dict[str, float]:
    """
    Calculate balanced class weights using training labels only.

    The validation and test sets are never used here.
    """

    classes = np.array(
        sorted(
            y_train.dropna().unique()
        )
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {
        str(class_name): float(weight)
        for class_name, weight in zip(
            classes,
            weights,
        )
    }


def create_sample_weights(
    y_train: pd.Series,
) -> pd.Series:
    """
    Create one sample weight for every training record.
    """

    class_weights = calculate_class_weights(
        y_train
    )

    return y_train.map(
        class_weights
    ).astype(float)