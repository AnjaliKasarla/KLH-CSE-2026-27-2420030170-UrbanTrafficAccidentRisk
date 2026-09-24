"""
Temporal feature engineering for the Urban Traffic Accident project.
"""

from __future__ import annotations

import pandas as pd


def add_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add model-ready temporal features.

    Expected input columns:
        Accident Date
        Time
        Day_of_Week
        Month
        Year
    """

    processed_df = df.copy()

    # ---------------------------------------------------------------
    # Accident date
    # ---------------------------------------------------------------

    if "Accident Date" in processed_df.columns:
        accident_date = pd.to_datetime(
            processed_df["Accident Date"],
            errors="coerce",
        )

        processed_df["Accident_Day"] = (
            accident_date.dt.day
        )

        processed_df["Accident_Month_Num"] = (
            accident_date.dt.month
        )

        processed_df["Accident_Week"] = (
            accident_date.dt.isocalendar()
            .week
            .astype("Int64")
        )

        processed_df["Accident_Day_Of_Year"] = (
            accident_date.dt.dayofyear
        )

    # ---------------------------------------------------------------
    # Time
    # ---------------------------------------------------------------

    if "Time" in processed_df.columns:
        parsed_time = pd.to_datetime(
            processed_df["Time"],
            format="%H:%M",
            errors="coerce",
        )

        processed_df["Accident_Hour"] = (
            parsed_time.dt.hour.astype("Int64")
        )

        processed_df["Accident_Minute"] = (
            parsed_time.dt.minute.astype("Int64")
        )

        processed_df["Is_Peak_Hour"] = (
            processed_df["Accident_Hour"]
            .isin([7, 8, 9, 16, 17, 18, 19])
            .astype("int8")
        )

    # ---------------------------------------------------------------
    # Cyclic time representation
    # ---------------------------------------------------------------

    if "Accident_Hour" in processed_df.columns:
        import numpy as np

        hour = (
            processed_df["Accident_Hour"]
            .fillna(-1)
            .astype(float)
        )

        valid_hour = hour >= 0

        processed_df["Hour_Sin"] = 0.0
        processed_df["Hour_Cos"] = 0.0

        processed_df.loc[valid_hour, "Hour_Sin"] = np.sin(
            2 * np.pi * hour[valid_hour] / 24
        )

        processed_df.loc[valid_hour, "Hour_Cos"] = np.cos(
            2 * np.pi * hour[valid_hour] / 24
        )

    return processed_df