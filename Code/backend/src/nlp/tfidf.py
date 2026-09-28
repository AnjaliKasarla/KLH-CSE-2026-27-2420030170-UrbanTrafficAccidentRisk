"""
TF-IDF baseline for the Urban Traffic Accident project.

This module converts the derived accident-context text into a sparse
TF-IDF representation.

The text is a structured-context representation generated from the
dataset attributes; it is not an original accident narrative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


DEFAULT_MAX_FEATURES = 5000
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 2
DEFAULT_MAX_DF = 0.95


class TfidfEncoder:
    """TF-IDF encoder for accident-context text."""

    def __init__(
        self,
        max_features: int = DEFAULT_MAX_FEATURES,
        ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
        min_df: int = DEFAULT_MIN_DF,
        max_df: float = DEFAULT_MAX_DF,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
        )

    def fit_transform(
        self,
        texts: list[str],
    ) -> csr_matrix:
        """Fit the vectorizer and transform text into TF-IDF features."""

        return self.vectorizer.fit_transform(texts)

    def transform(
        self,
        texts: list[str],
    ) -> csr_matrix:
        """Transform text using an already fitted vectorizer."""

        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> np.ndarray:
        """Return the learned TF-IDF feature names."""

        return self.vectorizer.get_feature_names_out()

    def save(self, output_path: str | Path) -> None:
        """Save the fitted TF-IDF vectorizer."""

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.vectorizer,
            output_path,
        )

    @classmethod
    def load(
        cls,
        input_path: str | Path,
    ) -> "TfidfEncoder":
        """Load a previously fitted TF-IDF vectorizer."""

        encoder = cls()
        encoder.vectorizer = joblib.load(input_path)

        return encoder