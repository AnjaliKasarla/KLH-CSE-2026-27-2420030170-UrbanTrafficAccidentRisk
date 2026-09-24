"""
XGBoost model for Urban Traffic Accident Severity prediction.
"""

from __future__ import annotations

from xgboost import XGBClassifier


def create_xgboost() -> XGBClassifier:
    """
    Create the XGBoost multiclass classifier.

    XGBoost requires integer class labels for multiclass
    classification.
    """

    return XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )