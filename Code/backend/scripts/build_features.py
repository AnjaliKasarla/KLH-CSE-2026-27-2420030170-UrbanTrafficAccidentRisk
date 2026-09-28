"""
Build the engineered feature dataset for the Urban Traffic Accident project.

Pipeline:
Raw CSV
    ↓
Missing-value handling
    ↓
Structured preprocessing
    ↓
Temporal features
    ↓
Weather features
    ↓
Road features
    ↓
Location features
    ↓
Leakage-safe feature/target split
    ↓
Processed model dataset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

from src.features.fusion import build_model_dataset
from src.preprocessing.missing_values import handle_missing_values
from src.preprocessing.structured import preprocess_structured_data


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "road_accident"
    / "Road Accident Data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final"
)

FEATURES_PATH = OUTPUT_DIR / "features.csv"
TARGET_PATH = OUTPUT_DIR / "target.csv"
METADATA_PATH = OUTPUT_DIR / "feature_metadata.json"


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main() -> None:
    """Build and save the final engineered feature dataset."""

    print("=" * 70)
    print("URBAN TRAFFIC ACCIDENT - FEATURE ENGINEERING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------

    print("\n[1/5] Loading raw dataset...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Raw rows    : {len(df):,}")
    print(f"Raw columns : {len(df.columns)}")

    # ---------------------------------------------------------------
    # 2. Missing-value handling
    # ---------------------------------------------------------------

    print("\n[2/5] Handling missing values...")

    df = handle_missing_values(df)

    # ---------------------------------------------------------------
    # 3. Structured preprocessing
    # ---------------------------------------------------------------

    print("\n[3/5] Applying structured preprocessing...")

    df = preprocess_structured_data(df)

    print(f"Rows after preprocessing    : {len(df):,}")
    print(f"Columns after preprocessing : {len(df.columns)}")

    # ---------------------------------------------------------------
    # 4. Feature engineering
    # ---------------------------------------------------------------

    print("\n[4/5] Building engineered features...")

    X, y = build_model_dataset(df)

    print(f"Feature rows  : {len(X):,}")
    print(f"Feature count : {len(X.columns)}")
    print(f"Target rows   : {len(y):,}")

    # ---------------------------------------------------------------
    # 5. Save outputs
    # ---------------------------------------------------------------

    print("\n[5/5] Saving processed datasets...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X.to_csv(
        FEATURES_PATH,
        index=False,
    )

    y.to_csv(
        TARGET_PATH,
        index=False,
        header=True,
    )

    metadata = {
        "source_dataset": str(
            RAW_DATA_PATH.relative_to(PROJECT_ROOT)
        ),
        "rows": int(len(X)),
        "feature_count": int(len(X.columns)),
        "target_column": "Accident_Severity",
        "target_classes": sorted(
            y.dropna().unique().tolist()
        ),
        "target_distribution": {
            str(key): int(value)
            for key, value in y.value_counts().to_dict().items()
        },
        "excluded_columns": [
            "Accident_Index",
            "Accident_Severity",
            "Number_of_Casualties",
        ],
        "feature_columns": X.columns.tolist(),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    # ---------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------

    assert len(X) == len(y)

    assert "Accident_Severity" not in X.columns

    assert "Number_of_Casualties" not in X.columns

    assert "Accident_Index" not in X.columns

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"Features : {FEATURES_PATH}")
    print(f"Target   : {TARGET_PATH}")
    print(f"Metadata : {METADATA_PATH}")

    print("\nFinal dataset:")
    print(f"Rows     : {len(X):,}")
    print(f"Features : {len(X.columns)}")

    print("\nTarget distribution:")

    print(
        y.value_counts()
        .rename_axis("Accident_Severity")
        .to_string()
    )

    print("\nLeakage checks:")
    print("Accident_Severity       :", "Accident_Severity" not in X.columns)
    print("Number_of_Casualties    :", "Number_of_Casualties" not in X.columns)
    print("Accident_Index          :", "Accident_Index" not in X.columns)


if __name__ == "__main__":
    main()