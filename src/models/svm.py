"""
Support Vector Machine model for Urban Traffic Accident Severity prediction.
"""

from __future__ import annotations

from sklearn.svm import SVC


def create_svm() -> SVC:
    """
    Create the SVM classifier.

    Class imbalance is handled using class_weight.
    """

    return SVC(
        C=1.0,
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )