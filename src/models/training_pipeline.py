"""
Training data preparation pipeline for the Urban Traffic Accident project.

Responsibilities:
    1. Load engineered features and target.
    2. Split data into train, validation, and test sets.
    3. Identify numerical and categorical features.
    4. Handle missing values without data leakage.
    5. Encode categorical variables.
    6. Scale numerical variables.

Model training is intentionally kept separate from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

TARGET_COLUMN = "Accident_Severity"


# -------------------------------------------------------------------
# Data container
# -------------------------------------------------------------------

@dataclass
class DatasetSplits:
    """Container for train, validation, and test datasets."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


# -------------------------------------------------------------------
# Loading
# -------------------------------------------------------------------

def load_engineered_data(
    features_path: str | Path,
    target_path: str | Path,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the engineered feature matrix and target vector.
    """

    features_path = Path(features_path)
    target_path = Path(target_path)

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}"
        )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Target file not found: {target_path}"
        )

    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze("columns")

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    if y.name != TARGET_COLUMN:
        y.name = TARGET_COLUMN

    return X, y


# -------------------------------------------------------------------
# Feature identification
# -------------------------------------------------------------------

def identify_feature_types(
    X: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical feature columns.
    """

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    return numerical_columns, categorical_columns


# -------------------------------------------------------------------
# Dataset splitting
# -------------------------------------------------------------------

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> DatasetSplits:
    """
    Create stratified train, validation, and test splits.

    Final proportions:
        Train      : 60%
        Validation : 20%
        Test       : 20%
    """

    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    validation_fraction = (
        VALIDATION_SIZE
        / (1.0 - TEST_SIZE)
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_fraction,
            random_state=RANDOM_STATE,
            stratify=y_train_val,
        )
    )

    return DatasetSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------

def create_preprocessor(
    numerical_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    """
    Create a leakage-safe preprocessing transformer.

    Numerical features:
        Median imputation
        ↓
        Standard scaling

    Categorical features:
        Most-frequent imputation
        ↓
        One-hot encoding

    The transformer is fitted only on the training split.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# -------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------

def validate_splits(
    splits: DatasetSplits,
) -> None:
    """Validate split sizes and feature-target alignment."""

    total_rows = (
        len(splits.X_train)
        + len(splits.X_validation)
        + len(splits.X_test)
    )

    if total_rows <= 0:
        raise ValueError(
            "Dataset contains no rows after splitting."
        )

    if len(splits.X_train) != len(splits.y_train):
        raise ValueError(
            "Training features and target mismatch."
        )

    if len(splits.X_validation) != len(
        splits.y_validation
    ):
        raise ValueError(
            "Validation features and target mismatch."
        )

    if len(splits.X_test) != len(
        splits.y_test
    ):
        raise ValueError(
            "Test features and target mismatch."
        )


# -------------------------------------------------------------------
# Complete preparation
# -------------------------------------------------------------------

def prepare_training_data(
    features_path: str | Path,
    target_path: str | Path,
):
    """
    Load, split, and prepare the training data.

    Returns:
        splits
        preprocessor
        numerical_columns
        categorical_columns
    """

    X, y = load_engineered_data(
        features_path,
        target_path,
    )

    if TARGET_COLUMN in X.columns:
        raise ValueError(
            "Target leakage detected: "
            "target column exists in X."
        )

    numerical_columns, categorical_columns = (
        identify_feature_types(X)
    )

    splits = split_dataset(
        X,
        y,
    )

    validate_splits(
        splits
    )

    preprocessor = create_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    return (
        splits,
        preprocessor,
        numerical_columns,
        categorical_columns,
    )