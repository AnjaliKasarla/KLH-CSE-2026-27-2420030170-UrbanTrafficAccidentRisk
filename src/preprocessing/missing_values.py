"""
Missing-value handling for the Urban Traffic Accident dataset.

The raw dataset is never modified directly. Missing values are handled
on a copy of the dataframe during preprocessing.
"""

from __future__ import annotations

import pandas as pd


CATEGORICAL_MISSING_VALUE = "Unknown"


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the accident dataset.

    Categorical columns are filled with an explicit 'Unknown' category.
    Numerical columns are preserved for downstream processing unless
    their handling is explicitly defined by the modeling pipeline.
    """

    cleaned_df = df.copy()

    categorical_columns = [
        "Carriageway_Hazards",
        "Road_Surface_Conditions",
        "Road_Type",
        "Weather_Conditions",
    ]

    for column in categorical_columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = cleaned_df[column].fillna(
                CATEGORICAL_MISSING_VALUE
            )

    if "Time" in cleaned_df.columns:
        cleaned_df["Time"] = cleaned_df["Time"].fillna(
            CATEGORICAL_MISSING_VALUE
        )

    return cleaned_df
