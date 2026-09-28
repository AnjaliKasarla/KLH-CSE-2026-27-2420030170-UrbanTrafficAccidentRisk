"""
Structured preprocessing for the Urban Traffic Accident dataset.

Handles duplicate records, date/time parsing, temporal feature
extraction, and removal of non-predictive identifiers.
"""

from __future__ import annotations

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate records."""

    return df.drop_duplicates().reset_index(drop=True)


def parse_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse accident date and create derived temporal features.
    """

    processed_df = df.copy()

    if "Accident Date" in processed_df.columns:
        processed_df["Accident Date"] = pd.to_datetime(
            processed_df["Accident Date"],
            errors="coerce",
        )

        processed_df["Accident_Day"] = (
            processed_df["Accident Date"].dt.day
        )

        processed_df["Accident_Month_Num"] = (
            processed_df["Accident Date"].dt.month
        )

        processed_df["Accident_Week"] = (
            processed_df["Accident Date"]
            .dt.isocalendar()
            .week
            .astype("Int64")
        )

    return processed_df


def parse_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse accident time and create hour-based temporal features.

    Missing time values are preserved as unknown rather than being
    artificially imputed.
    """

    processed_df = df.copy()

    if "Time" not in processed_df.columns:
        return processed_df

    parsed_time = pd.to_datetime(
        processed_df["Time"],
        format="%H:%M",
        errors="coerce",
    )

    processed_df["Accident_Hour"] = (
        parsed_time.dt.hour.astype("Int64")
    )

    processed_df["Time_Period"] = pd.cut(
        processed_df["Accident_Hour"],
        bins=[-1, 5, 11, 17, 21, 24],
        labels=[
            "Night",
            "Morning",
            "Afternoon",
            "Evening",
            "Late_Night",
        ],
    )

    processed_df["Time_Period"] = (
        processed_df["Time_Period"]
        .astype("string")
        .fillna("Unknown")
    )

    return processed_df


def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier fields that should not be used as predictive
    features.
    """

    processed_df = df.copy()

    identifier_columns = [
        "Accident_Index",
    ]

    existing_columns = [
        column
        for column in identifier_columns
        if column in processed_df.columns
    ]

    return processed_df.drop(
        columns=existing_columns
    )


def preprocess_structured_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the structured preprocessing pipeline.
    """

    processed_df = df.copy()

    processed_df = remove_duplicates(processed_df)
    processed_df = parse_date_features(processed_df)
    processed_df = parse_time_features(processed_df)
    processed_df = remove_identifier_columns(processed_df)

    return processed_df
