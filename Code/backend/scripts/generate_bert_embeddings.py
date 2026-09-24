"""
Generate production DistilBERT embeddings for the Urban Traffic
Accident dataset.

The pipeline:
1. Loads the raw accident dataset.
2. Applies missing-value handling.
3. Applies structured preprocessing.
4. Creates derived accident-context text.
5. Generates DistilBERT embeddings in batches.
6. Stores embeddings incrementally using NumPy memmap.
7. Saves the processed row count and embedding metadata.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.missing_values import (
    handle_missing_values,
)

from src.preprocessing.structured import (
    preprocess_structured_data,
)

from src.nlp.text_features import (
    add_accident_context_text,
)

from src.nlp.bert_embeddings import (
    save_dataframe_embeddings,
    EMBEDDING_DIMENSION,
)


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "road_accident"
    / "Road Accident Data.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "text"
    / "bert_embeddings.dat"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "text"
    / "bert_embeddings_metadata.json"
)

BATCH_SIZE = 8


def main() -> None:

    print("=" * 60)
    print("URBAN TRAFFIC ACCIDENT — BERT EMBEDDING GENERATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load raw data
    # --------------------------------------------------

    print("\n[1/6] Loading raw dataset...")

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"Raw rows: {len(df):,}"
    )

    # --------------------------------------------------
    # 2. Missing-value handling
    # --------------------------------------------------

    print(
        "\n[2/6] Handling missing values..."
    )

    df = handle_missing_values(df)

    # --------------------------------------------------
    # 3. Structured preprocessing
    # --------------------------------------------------

    print(
        "\n[3/6] Running structured preprocessing..."
    )

    df = preprocess_structured_data(df)

    print(
        f"Processed rows: {len(df):,}"
    )

    # --------------------------------------------------
    # 4. Create derived text
    # --------------------------------------------------

    print(
        "\n[4/6] Creating accident-context text..."
    )

    df = add_accident_context_text(
        df
    )

    print(
        "Text column created:",
        "Accident_Context_Text" in df.columns,
    )

    # --------------------------------------------------
    # 5. Generate embeddings
    # --------------------------------------------------

    print(
        "\n[5/6] Generating DistilBERT embeddings..."
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIMENSION}"
    )

    save_dataframe_embeddings(
        df=df,
        output_path=OUTPUT_PATH,
        text_column="Accident_Context_Text",
        batch_size=BATCH_SIZE,
    )

    # --------------------------------------------------
    # 6. Save metadata
    # --------------------------------------------------

    print(
        "\n[6/6] Saving metadata..."
    )

    metadata = {
        "model": "distilbert-base-uncased",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "dtype": "float32",
        "batch_size": BATCH_SIZE,
        "rows": len(df),
        "source": str(
            RAW_DATA_PATH
        ),
        "output": str(
            OUTPUT_PATH
        ),
        "text_column": "Accident_Context_Text",
    }

    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    import json

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

    print(
        "\n" + "=" * 60
    )

    print(
        "BERT EMBEDDING GENERATION COMPLETED"
    )

    print(
        f"Rows processed : {len(df):,}"
    )

    print(
        f"Embeddings     : {OUTPUT_PATH}"
    )

    print(
        f"Metadata       : {METADATA_PATH}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()