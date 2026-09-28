"""
Decision Tree model for Urban Traffic Accident Severity prediction.
"""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier


def create_decision_tree() -> DecisionTreeClassifier:
    """
    Create the Decision Tree classifier.

    Class imbalance is handled using class_weight.
    """

    return DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    )