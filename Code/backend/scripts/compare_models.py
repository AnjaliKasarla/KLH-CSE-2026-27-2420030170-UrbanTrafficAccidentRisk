"""
Compare validation metrics across trained accident-severity models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

OUTPUT_PATH = (
    METRICS_DIR
    / "model_comparison.csv"
)


MODEL_FILES = {
    "Logistic Regression": "logistic_regression_validation.json",
    "Decision Tree": "decision_tree_validation.json",
    "Random Forest": "random_forest_validation.json",
    "XGBoost": "xgboost_validation.json",
}


def load_metrics() -> pd.DataFrame:
    """Load validation metrics from all completed models."""

    rows = []

    for model_name, filename in MODEL_FILES.items():

        path = METRICS_DIR / filename

        if not path.exists():
            print(
                f"WARNING: Missing metrics file: {filename}"
            )
            continue

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "Macro Precision": metrics.get(
                    "macro_precision"
                ),
                "Macro Recall": metrics.get(
                    "macro_recall"
                ),
                "Macro F1": metrics.get(
                    "macro_f1"
                ),
                "Weighted F1": metrics.get(
                    "weighted_f1"
                ),
                "ROC-AUC": metrics.get(
                    "roc_auc"
                ),
                "Training Time (s)": metrics.get(
                    "training_time_seconds"
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Generate model comparison CSV."""

    print("=" * 70)
    print("MODEL VALIDATION COMPARISON")
    print("=" * 70)

    dataframe = load_metrics()

    if dataframe.empty:
        raise SystemExit(
            "No model metrics were found."
        )

    dataframe = dataframe.round(4)

    print("\n")
    print(dataframe.to_string(index=False))

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "-" * 70)
    print(
        f"Comparison saved to:\n{OUTPUT_PATH}"
    )
    print("-" * 70)


if __name__ == "__main__":
    main()