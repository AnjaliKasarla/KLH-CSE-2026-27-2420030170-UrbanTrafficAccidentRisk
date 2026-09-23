
"""
Categorical encoding for the Urban Traffic Accident dataset.

Uses one-hot encoding for nominal categorical features.
The target variable is kept separate from feature encoding.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "Accident_Severity"


def identify_categorical_columns(
    df: pd.DataFrame,
) -> List[str]:
    """Return categorical feature columns excluding the target."""

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    if TARGET_COLUMN in categorical_columns:
        categorical_columns.remove(TARGET_COLUMN)

    return categorical_columns


def create_encoder(
    df: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Create a one-hot encoder for categorical features.

    Returns:
        encoder: Configured ColumnTransformer.
        categorical_columns: Columns selected for encoding.
    """

    categorical_columns = identify_categorical_columns(df)

    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                categorical_columns,
            )
        ],
        remainder="passthrough",
    )

    return encoder, categorical_columns


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features from the target variable.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN].copy()

    return X, y