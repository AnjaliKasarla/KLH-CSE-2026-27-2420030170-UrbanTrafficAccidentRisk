"""Tests for accident feature engineering."""

import numpy as np
import pandas as pd

from src.features.location import add_location_features
from src.features.road import add_road_features
from src.features.temporal import add_temporal_features
from src.features.weather import add_weather_features


def test_temporal_features():
    df = pd.DataFrame(
        {
            "Accident Date": ["2026-09-25"],
            "Time": ["18:30"],
            "Day_of_Week": ["Friday"],
            "Month": ["September"],
            "Year": [2026],
        }
    )

    result = add_temporal_features(df)

    assert result.loc[0, "Accident_Day"] == 25
    assert result.loc[0, "Accident_Month_Num"] == 9
    assert result.loc[0, "Accident_Hour"] == 18
    assert result.loc[0, "Accident_Minute"] == 30
    assert result.loc[0, "Is_Peak_Hour"] == 1

    assert "Accident_Week" in result.columns
    assert "Accident_Day_Of_Year" in result.columns
    assert "Hour_Sin" in result.columns
    assert "Hour_Cos" in result.columns

    assert np.isfinite(result.loc[0, "Hour_Sin"])
    assert np.isfinite(result.loc[0, "Hour_Cos"])


def test_temporal_cyclic_features():
    df = pd.DataFrame(
        {
            "Time": ["00:00", "06:00", "12:00", "18:00"],
        }
    )

    result = add_temporal_features(df)

    assert np.isclose(result.loc[0, "Hour_Sin"], 0.0)
    assert np.isclose(result.loc[0, "Hour_Cos"], 1.0)

    assert np.isclose(result.loc[2, "Hour_Sin"], 0.0, atol=1e-6)
    assert np.isclose(result.loc[2, "Hour_Cos"], -1.0, atol=1e-6)


def test_weather_features():
    df = pd.DataFrame(
        {
            "Weather_Conditions": [
                "Fine no high winds",
                "Raining no high winds",
                "Snowing without high winds",
                "Fog or mist",
            ],
            "Road_Surface_Conditions": [
                "Dry",
                "Wet or damp",
                "Snow",
                "Frost or ice",
            ],
            "Light_Conditions": [
                "Daylight",
                "Darkness - lights lit",
                "Darkness - lights unlit",
                "Daylight",
            ],
        }
    )

    result = add_weather_features(df)

    assert result["Weather_Category"].tolist() == [
        "Clear",
        "Rain",
        "Snow",
        "Fog_Mist",
    ]

    assert result["Road_Surface_Category"].tolist() == [
        "Dry",
        "Wet",
        "Snow_Ice",
        "Snow_Ice",
    ]

    assert result["Light_Category"].tolist() == [
        "Daylight",
        "Dark",
        "Dark",
        "Daylight",
    ]


def test_weather_unknown_values():
    df = pd.DataFrame(
        {
            "Weather_Conditions": [None],
            "Road_Surface_Conditions": [None],
            "Light_Conditions": [None],
        }
    )

    result = add_weather_features(df)

    assert result.loc[0, "Weather_Category"] == "Unknown"
    assert result.loc[0, "Road_Surface_Category"] == "Unknown"
    assert result.loc[0, "Light_Category"] == "Unknown"


def test_road_features():
    df = pd.DataFrame(
        {
            "Speed_limit": [30, 50, 70, 100],
            "Number_of_Vehicles": [1, 2, 4, 5],
            "Junction_Detail": [
                "Not at junction",
                "Roundabout",
                "Crossroads",
                None,
            ],
            "Road_Type": [
                "Motorway",
                "Dual carriageway",
                "Single carriageway",
                None,
            ],
            "Vehicle_Type": [
                "Car",
                "Motorcycle",
                "Bus or coach",
                None,
            ],
        }
    )

    result = add_road_features(df)

    assert result["Speed_Limit_Category"].tolist() == [
        "Low",
        "Medium",
        "High",
        "Very_High",
    ]

    assert result["Vehicle_Count_Category"].tolist() == [
        "Single",
        "Low",
        "Medium",
        "High",
    ]

    assert result.loc[0, "Junction_Type"] == "No_Junction"
    assert result.loc[1, "Junction_Type"] == "roundabout"

    assert result.loc[0, "Road_Category"] == "Motorway"
    assert result.loc[1, "Road_Category"] == "Dual_Carriageway"
    assert result.loc[2, "Road_Category"] == "Single_Carriageway"
    assert result.loc[3, "Road_Category"] == "Unknown"

    assert result.loc[0, "Vehicle_Category"] == "Car"
    assert result.loc[1, "Vehicle_Category"] == "Motorcycle"
    assert result.loc[2, "Vehicle_Category"] == "Bus_Coach"
    assert result.loc[3, "Vehicle_Category"] == "Unknown"


def test_location_features():
    df = pd.DataFrame(
        {
            "Latitude": [17.385],
            "Longitude": [78.4867],
            "Urban_or_Rural_Area": ["Urban"],
            "Local_Authority_(District)": ["Hyderabad"],
            "Police_Force": ["Hyderabad"],
        }
    )

    result = add_location_features(df)

    assert np.isclose(
        result.loc[0, "Lat_Long_Interaction"],
        17.385 * 78.4867,
    )

    assert result.loc[0, "Area_Category"] == "Urban"
    assert result.loc[0, "Local_Authority_(District)_Group"] == "Hyderabad"
    assert result.loc[0, "Police_Force_Group"] == "Hyderabad"

    assert "Latitude_Region" in result.columns
    assert "Longitude_Region" in result.columns


def test_location_numeric_conversion():
    df = pd.DataFrame(
        {
            "Latitude": ["17.385"],
            "Longitude": ["78.4867"],
        }
    )

    result = add_location_features(df)

    assert result["Latitude"].dtype.kind in "fc"
    assert result["Longitude"].dtype.kind in "fc"


def test_feature_functions_preserve_original_columns():
    df = pd.DataFrame(
        {
            "Accident Date": ["2026-09-25"],
            "Time": ["18:30"],
            "Weather_Conditions": ["Fine no high winds"],
            "Road_Type": ["Single carriageway"],
            "Latitude": [17.385],
            "Longitude": [78.4867],
        }
    )

    temporal = add_temporal_features(df)
    weather = add_weather_features(df)
    road = add_road_features(df)
    location = add_location_features(df)

    for result in [temporal, weather, road, location]:
        for column in df.columns:
            assert column in result.columns
