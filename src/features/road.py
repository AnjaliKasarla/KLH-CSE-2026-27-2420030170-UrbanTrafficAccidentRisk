"""
Road and traffic feature engineering for the
Urban Traffic Accident project.
"""

from __future__ import annotations

import pandas as pd


def add_road_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add road, junction, speed, and traffic-related features.

    Expected input columns:
        Road_Type
        Junction_Control
        Junction_Detail
        Speed_limit
        Number_of_Vehicles
        Number_of_Casualties
        Vehicle_Type
    """

    processed_df = df.copy()

    # ---------------------------------------------------------------
    # Speed-limit categories
    # ---------------------------------------------------------------

    if "Speed_limit" in processed_df.columns:

        speed = pd.to_numeric(
            processed_df["Speed_limit"],
            errors="coerce",
        )

        processed_df["Speed_Limit_Category"] = pd.cut(
            speed,
            bins=[
                -float("inf"),
                30,
                50,
                70,
                float("inf"),
            ],
            labels=[
                "Low",
                "Medium",
                "High",
                "Very_High",
            ],
        ).astype("string").fillna("Unknown")

    # ---------------------------------------------------------------
    # Vehicle count categories
    # ---------------------------------------------------------------

    if "Number_of_Vehicles" in processed_df.columns:

        vehicles = pd.to_numeric(
            processed_df["Number_of_Vehicles"],
            errors="coerce",
        )

        processed_df["Vehicle_Count_Category"] = pd.cut(
            vehicles,
            bins=[
                -float("inf"),
                1,
                2,
                4,
                float("inf"),
            ],
            labels=[
                "Single",
                "Low",
                "Medium",
                "High",
            ],
        ).astype("string").fillna("Unknown")

    # ---------------------------------------------------------------
    # Junction complexity
    # ---------------------------------------------------------------

    if "Junction_Detail" in processed_df.columns:

        junction = (
            processed_df["Junction_Detail"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        processed_df["Junction_Type"] = (
            junction
            .replace({
                "not at junction": "No_Junction",
                "not at junction or within 20 metres": "No_Junction",
            })
        )

    # ---------------------------------------------------------------
    # Road-type grouping
    # ---------------------------------------------------------------

    if "Road_Type" in processed_df.columns:

        road_type = (
            processed_df["Road_Type"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        processed_df["Road_Category"] = (
            road_type
            .map(_categorize_road)
            .fillna("Other")
        )

    # ---------------------------------------------------------------
    # Vehicle type grouping
    # ---------------------------------------------------------------

    if "Vehicle_Type" in processed_df.columns:

        vehicle_type = (
            processed_df["Vehicle_Type"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        processed_df["Vehicle_Category"] = (
            vehicle_type
            .map(_categorize_vehicle)
            .fillna("Other")
        )

    return processed_df


def _categorize_road(
    value: str,
) -> str:
    """Group road types into broader categories."""

    if value == "unknown":
        return "Unknown"

    if "motorway" in value:
        return "Motorway"

    if "dual carriageway" in value:
        return "Dual_Carriageway"

    if "single carriageway" in value:
        return "Single_Carriageway"

    if "roundabout" in value:
        return "Roundabout"

    return "Other"


def _categorize_vehicle(
    value: str,
) -> str:
    """Group vehicle types into broader categories."""

    if value == "unknown":
        return "Unknown"

    if "car" in value:
        return "Car"

    if "motorcycle" in value:
        return "Motorcycle"

    if "bus" in value or "coach" in value:
        return "Bus_Coach"

    if "goods" in value or "van" in value:
        return "Goods_Van"

    if "agricultural" in value:
        return "Agricultural"

    return "Other"