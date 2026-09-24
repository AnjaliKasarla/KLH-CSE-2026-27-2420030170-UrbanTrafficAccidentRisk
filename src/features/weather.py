"""
Weather and environmental feature engineering for the
Urban Traffic Accident project.
"""

from __future__ import annotations

import pandas as pd


def add_weather_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add weather/environment-related features.

    Expected input columns:
        Weather_Conditions
        Road_Surface_Conditions
        Light_Conditions
    """

    processed_df = df.copy()

    # ---------------------------------------------------------------
    # Weather severity grouping
    # ---------------------------------------------------------------

    if "Weather_Conditions" in processed_df.columns:

        weather = (
            processed_df["Weather_Conditions"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        processed_df["Weather_Category"] = (
            weather
            .map(_categorize_weather)
            .fillna("Other")
        )

    # ---------------------------------------------------------------
    # Road surface grouping
    # ---------------------------------------------------------------

    if "Road_Surface_Conditions" in processed_df.columns:

        surface = (
            processed_df["Road_Surface_Conditions"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        processed_df["Road_Surface_Category"] = (
            surface
            .map(_categorize_surface)
            .fillna("Other")
        )

    # ---------------------------------------------------------------
    # Light condition grouping
    # ---------------------------------------------------------------

    if "Light_Conditions" in processed_df.columns:

        light = (
            processed_df["Light_Conditions"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        processed_df["Light_Category"] = (
            light
            .map(_categorize_light)
            .fillna("Other")
        )

    return processed_df


def _categorize_weather(
    value: str,
) -> str:
    """Group raw weather conditions into broader categories."""

    if value == "unknown":
        return "Unknown"

    if "rain" in value or "drizzle" in value:
        return "Rain"

    if "snow" in value:
        return "Snow"

    if "fog" in value or "mist" in value:
        return "Fog_Mist"

    if "fine" in value or "clear" in value:
        return "Clear"

    if "wind" in value or "storm" in value:
        return "Wind_Storm"

    return "Other"


def _categorize_surface(
    value: str,
) -> str:
    """Group road-surface conditions."""

    if value == "unknown":
        return "Unknown"

    if "wet" in value:
        return "Wet"

    if "snow" in value or "frost" in value or "ice" in value:
        return "Snow_Ice"

    if "dry" in value:
        return "Dry"

    return "Other"


def _categorize_light(
    value: str,
) -> str:
    """Group lighting conditions."""

    if value == "unknown":
        return "Unknown"

    if "dark" in value:
        return "Dark"

    if "daylight" in value:
        return "Daylight"

    return "Other"