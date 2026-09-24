"""
Generate the TF-IDF baseline representation for the
Urban Traffic Accident project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# -------------------------------------------------------------------
# Project path configuration
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------------------------
# Third-party imports
# -------------------------------------------------------------------

import pandas as pd
from scipy.sparse import save_npz

# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

from src.nlp.text_features import add_accident_context_text
from src.nlp.tfidf import TfidfEncoder
from src.preprocessing.missing_values import handle_missing_values
from src.preprocessing.structured import preprocess_structured_data


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

INPUT_PATH = (
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
    / "text"
)

MATRIX_PATH = OUTPUT_DIR / "tfidf_matrix.npz"
VECTORIZER_PATH = OUTPUT_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = OUTPUT_DIR / "tfidf_metadata.json"


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("URBAN TRAFFIC ACCIDENT — TF-IDF BASELINE")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------

    print("\n[1/5] Loading dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Raw rows: {len(df):,}")

    # ---------------------------------------------------------------
    # 2. Preprocessing
    # ---------------------------------------------------------------

    print("\n[2/5] Preprocessing...")

    df = handle_missing_values(df)
    df = preprocess_structured_data(df)

    print(f"Processed rows: {len(df):,}")

    # ---------------------------------------------------------------
    # 3. Create derived text
    # ---------------------------------------------------------------

    print("\n[3/5] Creating accident-context text...")

    df = add_accident_context_text(df)

    texts = (
        df["Accident_Context_Text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(f"Text records: {len(texts):,}")

    # ---------------------------------------------------------------
    # 4. TF-IDF transformation
    # ---------------------------------------------------------------

    print("\n[4/5] Fitting TF-IDF...")

    encoder = TfidfEncoder()

    matrix = encoder.fit_transform(texts)

    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero values: {matrix.nnz:,}")

    # ---------------------------------------------------------------
    # 5. Save artifacts
    # ---------------------------------------------------------------

    print("\n[5/5] Saving artifacts...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_npz(
        MATRIX_PATH,
        matrix,
    )

    encoder.save(
        VECTORIZER_PATH,
    )

    metadata = {
        "source": str(INPUT_PATH),
        "rows": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "matrix_format": "scipy_csr",
        "max_features": 5000,
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95,
        "text_column": "Accident_Context_Text",
        "representation": "TF-IDF",
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Completion
    # ---------------------------------------------------------------

    print("\n" + "=" * 60)
    print("TF-IDF BASELINE COMPLETED")
    print("=" * 60)

    print(f"Matrix    : {MATRIX_PATH}")
    print(f"Vectorizer: {VECTORIZER_PATH}")
    print(f"Metadata  : {METADATA_PATH}")


if __name__ == "__main__":
    main()