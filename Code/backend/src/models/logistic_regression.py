"""
Logistic Regression model for Urban Traffic Accident Severity prediction.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def create_logistic_regression() -> LogisticRegression:
    """
    Create the Logistic Regression classifier.

    Class imbalance is handled using class_weight.
    """

    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="saga",
        random_state=42,
    )