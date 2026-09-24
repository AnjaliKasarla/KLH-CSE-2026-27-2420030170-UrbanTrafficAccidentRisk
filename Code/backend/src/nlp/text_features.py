"""
Derived text representation for the Urban Traffic Accident dataset.

The source dataset does not contain a native free-text accident
description. This module therefore creates a structured-context
sentence from existing accident attributes.

The generated text is a derived representation and must not be
interpreted as an original accident narrative.
"""

from __future__ import annotations

import pandas as pd


TEXT_FIELDS = [
    "Junction_Control",
    "Junction_Detail",
    "Light_Conditions",
    "Road_Surface_Conditions",
    "Road_Type",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
    "Vehicle_Type",
    "Speed_limit",
]


def _clean_value(value: object) -> str:
    """Convert a dataset value into a safe text representation."""

    if pd.isna(value):
        return "unknown"

    return str(value).strip().lower()


def create_accident_context_text(
    row: pd.Series,
) -> str:
    """
    Create a derived accident-context sentence from one record.

    This is not an original accident description. It is a textual
    representation generated from structured dataset attributes.
    """

    parts = []

    for field in TEXT_FIELDS:
        if field not in row.index:
            continue

        value = _clean_value(row[field])

        field_name = (
            field
            .replace("_", " ")
            .replace("(", "")
            .replace(")", "")
        )

        parts.append(
            f"{field_name}: {value}"
        )

    return "; ".join(parts)


def add_accident_context_text(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a derived accident-context text column to the dataframe.
    """

    processed_df = df.copy()

    processed_df["Accident_Context_Text"] = (
        processed_df.apply(
            create_accident_context_text,
            axis=1,
        )
    )

    return processed_df