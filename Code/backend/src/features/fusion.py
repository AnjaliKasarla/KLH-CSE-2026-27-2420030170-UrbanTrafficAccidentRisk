"""
Feature fusion for the Urban Traffic Accident project.

Combines engineered temporal, weather, road, and location features
into a leakage-aware model dataset.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.features.location import add_location_features
from src.features.road import add_road_features
from src.features.temporal import add_temporal_features
from src.features.weather import add_weather_features


TARGET_COLUMN = "Accident_Severity"

# Columns unavailable or inappropriate for the initial
# pre-event accident-risk model.
EXCLUDED_MODEL_COLUMNS = {
    # Identifier
    "Accident_Index",

    # Target
    "Accident_Severity",

    # Raw temporal representations after feature extraction
    "Accident Date",
    "Time",
    "Month",
    "Day_of_Week",
    "Year",

    # Post-event information
    "Number_of_Casualties",
    "Number_of_Vehicles",
    "Vehicle_Count_Category",
}


def build_feature_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete engineered feature dataset."""

    processed_df = df.copy()

    processed_df = add_temporal_features(
        processed_df
    )

    processed_df = add_weather_features(
        processed_df
    )

    processed_df = add_road_features(
        processed_df
    )

    processed_df = add_location_features(
        processed_df
    )

    return processed_df


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate leakage-aware model features from the target."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    y = df[TARGET_COLUMN].copy()

    columns_to_drop = [
        column
        for column in EXCLUDED_MODEL_COLUMNS
        if column in df.columns
    ]

    X = df.drop(
        columns=columns_to_drop
    )

    return X, y


def build_model_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build engineered features and split into X and y."""

    engineered_df = build_feature_dataset(
        df
    )

    return split_features_target(
        engineered_df
    )