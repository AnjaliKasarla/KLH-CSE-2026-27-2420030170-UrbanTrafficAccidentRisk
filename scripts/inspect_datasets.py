"""
Dataset profiling for Urban Traffic Accident Risk Assessment.

Profiles the verified Road Accident dataset and produces
machine-readable JSON reports for Phase 1 dataset analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROAD_ACCIDENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "road_accident"
    / "Road Accident Data.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"


def profile_dataset(df: pd.DataFrame, dataset_name: str) -> dict:
    """Generate a structured dataset profile."""

    profile = {
        "dataset_name": dataset_name,
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": list(df.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },
        "missing_percent": {
            column: round(float((count / len(df)) * 100), 4)
            for column, count in df.isna().sum().items()
            if count > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(
            df.select_dtypes(include="number").columns
        ),
        "categorical_columns": list(
            df.select_dtypes(
                include=["object", "string", "category"]
            ).columns
        ),
        "unique_counts": {
            column: int(df[column].nunique(dropna=True))
            for column in df.columns
        },
    }

    return profile


def add_target_profile(
    profile: dict,
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """Add target distribution information when available."""

    if target_column not in df.columns:
        return

    counts = df[target_column].value_counts(dropna=False)

    percentages = (
        df[target_column]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        * 100
    )

    profile["target"] = {
        "column": target_column,
        "classes": {
            str(label): int(count)
            for label, count in counts.items()
        },
        "percentages": {
            str(label): round(float(value), 4)
            for label, value in percentages.items()
        },
        "missing": int(df[target_column].isna().sum()),
    }


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as formatted JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def main() -> None:
    print("=" * 70)
    print("URBAN TRAFFIC ACCIDENT RISK - DATASET PROFILING")
    print("=" * 70)

    if not ROAD_ACCIDENT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {ROAD_ACCIDENT_PATH}"
        )

    print(
        f"\nLoading dataset:\n"
        f"{ROAD_ACCIDENT_PATH}"
    )

    df = pd.read_csv(ROAD_ACCIDENT_PATH)

    print(f"\nDataset shape: {df.shape}")

    profile = profile_dataset(
        df=df,
        dataset_name="Road Accident Dataset",
    )

    add_target_profile(
        profile=profile,
        df=df,
        target_column="Accident_Severity",
    )

    output_path = (
        REPORTS_DIR
        / "road_accident_profile.json"
    )

    save_json(
        profile,
        output_path,
    )

    print("\n" + "-" * 70)
    print("DATASET SUMMARY")
    print("-" * 70)

    print(
        f"Rows              : "
        f"{df.shape[0]:,}"
    )

    print(
        f"Columns           : "
        f"{df.shape[1]}"
    )

    print(
        f"Duplicate rows    : "
        f"{df.duplicated().sum():,}"
    )

    print("\nMissing values:")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("  None")
    else:
        for column, count in missing.items():
            percentage = (
                count / len(df)
            ) * 100

            print(
                f"  {column}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

    print("\nTarget distribution:")

    target_counts = (
        df["Accident_Severity"]
        .value_counts()
    )

    for label, count in target_counts.items():
        percentage = (
            count / len(df)
        ) * 100

        print(
            f"  {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nNumeric columns:")

    for column in profile["numeric_columns"]:
        print(f"  - {column}")

    print("\nCategorical columns:")

    for column in profile["categorical_columns"]:
        print(f"  - {column}")

    print("\nProfile saved to:")
    print(output_path)

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()