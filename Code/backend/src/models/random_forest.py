"""
Random Forest model for Urban Traffic Accident Severity prediction.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def create_random_forest() -> RandomForestClassifier:
    """
    Create the Random Forest classifier.

    Class imbalance is handled using class_weight.
    """

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )