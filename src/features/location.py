"""
Location feature engineering for the Urban Traffic Accident project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_location_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add geographic and urban/rural location features.

    Expected input columns:
        Latitude
        Longitude
        Urban_or_Rural_Area
        Local_Authority_(District)
        Police_Force
    """

    processed_df = df.copy()

    # ---------------------------------------------------------------
    # Geographic coordinates
    # ---------------------------------------------------------------

    if "Latitude" in processed_df.columns:
        processed_df["Latitude"] = pd.to_numeric(
            processed_df["Latitude"],
            errors="coerce",
        )

    if "Longitude" in processed_df.columns:
        processed_df["Longitude"] = pd.to_numeric(
            processed_df["Longitude"],
            errors="coerce",
        )

    # ---------------------------------------------------------------
    # Coordinate interaction
    # ---------------------------------------------------------------

    if {
        "Latitude",
        "Longitude",
    }.issubset(processed_df.columns):

        processed_df["Lat_Long_Interaction"] = (
            processed_df["Latitude"]
            * processed_df["Longitude"]
        )

    # ---------------------------------------------------------------
    # Urban / rural category
    # ---------------------------------------------------------------

    if "Urban_or_Rural_Area" in processed_df.columns:

        urban_rural = (
            processed_df["Urban_or_Rural_Area"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        processed_df["Area_Category"] = (
            urban_rural
            .map(_categorize_area)
            .fillna("Other")
        )

    # ---------------------------------------------------------------
    # Latitude / longitude spatial bins
    # ---------------------------------------------------------------

    if "Latitude" in processed_df.columns:

        processed_df["Latitude_Region"] = pd.cut(
            processed_df["Latitude"],
            bins=10,
            labels=False,
        )

    if "Longitude" in processed_df.columns:

        processed_df["Longitude_Region"] = pd.cut(
            processed_df["Longitude"],
            bins=10,
            labels=False,
        )

    # ---------------------------------------------------------------
    # Local authority / police-force grouping
    # ---------------------------------------------------------------

    for column in [
        "Local_Authority_(District)",
        "Police_Force",
    ]:
        if column in processed_df.columns:

            processed_df[f"{column}_Group"] = (
                processed_df[column]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    return processed_df


def _categorize_area(
    value: str,
) -> str:
    """Normalize urban/rural area values."""

    if value == "unknown":
        return "Unknown"

    if "urban" in value:
        return "Urban"

    if "rural" in value:
        return "Rural"

    return "Other"