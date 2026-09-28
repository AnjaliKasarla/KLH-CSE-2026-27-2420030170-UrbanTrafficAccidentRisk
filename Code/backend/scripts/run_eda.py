"""
Exploratory Data Analysis for the Urban Traffic Accident Risk project.

Generates reproducible statistics and visualizations from the raw
Road Accident dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "road_accident"
    / "Road Accident Data.csv"
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TARGET_COLUMN = "Accident_Severity"

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def save_plot(filename: str) -> None:
    """Save and close the current matplotlib figure."""

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load the raw Road Accident dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df


# ---------------------------------------------------------------------
# Target analysis
# ---------------------------------------------------------------------

def analyze_target(df: pd.DataFrame) -> None:
    """Analyze accident severity distribution."""

    counts = df[TARGET_COLUMN].value_counts()

    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    target_summary = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    target_summary.to_csv(
        METRICS_DIR / "target_distribution.csv"
    )

    print("\nTARGET DISTRIBUTION")
    print(target_summary)

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        order=counts.index,
    )

    plt.title("Accident Severity Distribution")
    plt.xlabel("Accident Severity")
    plt.ylabel("Number of Accidents")

    save_plot(
        "01_accident_severity_distribution.png"
    )


# ---------------------------------------------------------------------
# General categorical analysis
# ---------------------------------------------------------------------

def analyze_categorical_feature(
    df: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
    top_n: int = 10,
) -> None:
    """Analyze a categorical feature against accident severity."""

    if column not in df.columns:
        return

    top_categories = (
        df[column]
        .value_counts()
        .head(top_n)
        .index
    )

    filtered_df = df[
        df[column].isin(top_categories)
    ]

    plt.figure(figsize=(11, 6))

    sns.countplot(
        data=filtered_df,
        y=column,
        hue=TARGET_COLUMN,
        order=top_categories,
    )

    plt.title(title)
    plt.xlabel("Number of Accidents")
    plt.ylabel(column)

    save_plot(filename)


# ---------------------------------------------------------------------
# Ordered categorical analysis
# ---------------------------------------------------------------------

def analyze_categorical_feature_with_order(
    df: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
    order: list[str],
) -> None:
    """Analyze an ordered categorical feature against severity."""

    if column not in df.columns:
        return

    plt.figure(figsize=(11, 6))

    sns.countplot(
        data=df,
        x=column,
        hue=TARGET_COLUMN,
        order=order,
    )

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)

    save_plot(filename)


# ---------------------------------------------------------------------
# Numerical analysis
# ---------------------------------------------------------------------

def analyze_numeric_feature(
    df: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
) -> None:
    """Analyze a numerical feature across accident severity."""

    if column not in df.columns:
        return

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y=column,
    )

    plt.title(title)
    plt.xlabel("Accident Severity")
    plt.ylabel(column)

    save_plot(filename)


def analyze_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
) -> None:
    """Analyze a numerical feature using severity-wise distributions."""

    analyze_numeric_feature(
        df,
        column,
        filename,
        title,
    )


# ---------------------------------------------------------------------
# Weather / road / environment analysis
# ---------------------------------------------------------------------

def analyze_environmental_features(
    df: pd.DataFrame,
) -> None:
    """Analyze environmental and road-condition features."""

    analyze_categorical_feature(
        df,
        "Weather_Conditions",
        "02_weather_vs_severity.png",
        "Weather Conditions vs Accident Severity",
    )

    analyze_categorical_feature(
        df,
        "Road_Surface_Conditions",
        "03_road_surface_vs_severity.png",
        "Road Surface Conditions vs Accident Severity",
    )

    analyze_categorical_feature(
        df,
        "Road_Type",
        "04_road_type_vs_severity.png",
        "Road Type vs Accident Severity",
    )

    analyze_categorical_feature(
        df,
        "Light_Conditions",
        "05_light_conditions_vs_severity.png",
        "Light Conditions vs Accident Severity",
    )

    analyze_categorical_feature(
        df,
        "Urban_or_Rural_Area",
        "06_urban_rural_vs_severity.png",
        "Urban/Rural Area vs Accident Severity",
    )


# ---------------------------------------------------------------------
# Speed analysis
# ---------------------------------------------------------------------

def analyze_speed_limit(df: pd.DataFrame) -> None:
    """Analyze speed limit against accident severity."""

    analyze_numeric_feature(
        df,
        "Speed_limit",
        "07_speed_limit_vs_severity.png",
        "Speed Limit vs Accident Severity",
    )


# ---------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------

def analyze_year(df: pd.DataFrame) -> None:
    """Analyze accident counts by year and severity."""

    if "Year" not in df.columns:
        return

    yearly = (
        df.groupby(
            ["Year", TARGET_COLUMN]
        )
        .size()
        .reset_index(name="Accident_Count")
    )

    yearly.to_csv(
        METRICS_DIR
        / "yearly_accident_distribution.csv",
        index=False,
    )

    plt.figure(figsize=(11, 6))

    sns.lineplot(
        data=yearly,
        x="Year",
        y="Accident_Count",
        hue=TARGET_COLUMN,
        marker="o",
    )

    plt.title("Accident Trends by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Accidents")

    save_plot(
        "08_accident_trends_by_year.png"
    )


def analyze_day_of_week(df: pd.DataFrame) -> None:
    """Analyze accident distribution by day of week."""

    if "Day_of_Week" not in df.columns:
        return

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    analyze_categorical_feature_with_order(
        df,
        "Day_of_Week",
        "10_day_of_week_vs_severity.png",
        "Day of Week vs Accident Severity",
        order,
    )


def analyze_month(df: pd.DataFrame) -> None:
    """Analyze accident distribution by month."""

    if "Month" not in df.columns:
        return

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    analyze_categorical_feature_with_order(
        df,
        "Month",
        "11_month_vs_severity.png",
        "Month vs Accident Severity",
        month_order,
    )


def analyze_accident_hour(df: pd.DataFrame) -> None:
    """Analyze accident distribution by hour."""

    if "Time" not in df.columns:
        return

    temp_df = df.copy()

    temp_df["Accident_Hour"] = pd.to_datetime(
        temp_df["Time"],
        format="%H:%M",
        errors="coerce",
    ).dt.hour

    temp_df = temp_df.dropna(
        subset=["Accident_Hour"]
    )

    temp_df["Accident_Hour"] = (
        temp_df["Accident_Hour"]
        .astype(int)
    )

    plt.figure(figsize=(14, 6))

    sns.countplot(
        data=temp_df,
        x="Accident_Hour",
        hue=TARGET_COLUMN,
    )

    plt.title(
        "Accident Hour vs Accident Severity"
    )
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Accidents")

    save_plot(
        "12_accident_hour_vs_severity.png"
    )


# ---------------------------------------------------------------------
# Casualty / vehicle analysis
# ---------------------------------------------------------------------

def analyze_casualties(df: pd.DataFrame) -> None:
    """Analyze number of casualties against severity."""

    analyze_numeric_distribution(
        df,
        "Number_of_Casualties",
        "13_casualties_vs_severity.png",
        "Number of Casualties vs Accident Severity",
    )


def analyze_vehicles(df: pd.DataFrame) -> None:
    """Analyze number of vehicles against severity."""

    analyze_numeric_distribution(
        df,
        "Number_of_Vehicles",
        "14_vehicles_vs_severity.png",
        "Number of Vehicles vs Accident Severity",
    )


# ---------------------------------------------------------------------
# Junction analysis
# ---------------------------------------------------------------------

def analyze_junction_features(
    df: pd.DataFrame,
) -> None:
    """Analyze junction characteristics against severity."""

    analyze_categorical_feature(
        df,
        "Junction_Control",
        "15_junction_control_vs_severity.png",
        "Junction Control vs Accident Severity",
    )

    analyze_categorical_feature(
        df,
        "Junction_Detail",
        "16_junction_detail_vs_severity.png",
        "Junction Detail vs Accident Severity",
    )


# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------

def analyze_numeric_correlations(
    df: pd.DataFrame,
) -> None:
    """Generate correlation matrix for numerical features."""

    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation = numeric_df.corr()

    correlation.to_csv(
        METRICS_DIR
        / "numeric_correlation_matrix.csv"
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )

    plt.title(
        "Numerical Feature Correlation Matrix"
    )

    save_plot(
        "09_numeric_correlation_matrix.png"
    )


# ---------------------------------------------------------------------
# Main EDA pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete EDA pipeline."""

    df = load_dataset()

    analyze_target(df)

    analyze_environmental_features(df)

    analyze_speed_limit(df)

    analyze_year(df)

    analyze_numeric_correlations(df)

    analyze_day_of_week(df)

    analyze_month(df)

    analyze_accident_hour(df)

    analyze_casualties(df)

    analyze_vehicles(df)

    analyze_junction_features(df)

    print("\n" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Figures saved to : {FIGURES_DIR}"
    )

    print(
        f"Metrics saved to : {METRICS_DIR}"
    )


if __name__ == "__main__":
    main()