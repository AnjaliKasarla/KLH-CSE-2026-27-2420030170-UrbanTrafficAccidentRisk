"""
Location feature engineering for the Urban Traffic Accident project.
"""

from __future__ import annotations

import pandas as pd


# Training-time spatial boundaries derived from the full
# Road Accident Data.csv dataset using pd.cut(..., bins=10).
LATITUDE_BIN_EDGES = [
    49.903804433,
    50.9828447,
    52.0512014,
    53.1195581,
    54.1879148,
    55.2562715,
    56.3246282,
    57.3929849,
    58.4613416,
    59.5296983,
    60.598055,
]

LONGITUDE_BIN_EDGES = [
    -7.525500623,
    -6.5886627,
    -5.6611004,
    -4.7335381,
    -3.8059758,
    -2.8784135,
    -1.9508512,
    -1.0232889,
    -0.0957266,
    0.8318357,
    1.759398,
]


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
    #
    # IMPORTANT:
    # These boundaries are fixed from the original training dataset.
    # They must not be recalculated from the inference dataframe,
    # because API inference may contain only one row.
    #

    if "Latitude" in processed_df.columns:
        processed_df["Latitude_Region"] = pd.cut(
            processed_df["Latitude"],
            bins=LATITUDE_BIN_EDGES,
            labels=False,
        )

    if "Longitude" in processed_df.columns:
        processed_df["Longitude_Region"] = pd.cut(
            processed_df["Longitude"],
            bins=LONGITUDE_BIN_EDGES,
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