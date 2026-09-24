import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.nlp.text_features import (
    add_accident_context_text,
)

from src.nlp.bert_embeddings import (
    save_dataframe_embeddings,
)


DATA_PATH = (
    "data/raw/road_accident/"
    "Road Accident Data.csv"
)

OUTPUT_PATH = (
    "data/interim/traffic_accidents/"
    "bert_embeddings_test.dat"
)

TEST_ROWS = 1024
BATCH_SIZE = 8


print("Loading test data...")

df = pd.read_csv(
    DATA_PATH,
    nrows=TEST_ROWS,
)

print("ROWS:", len(df))

print("\nCreating accident-context text...")

df = add_accident_context_text(df)

print(
    "TEXT COLUMN:",
    "Accident_Context_Text" in df.columns,
)

print("\nGenerating and saving embeddings...")

save_dataframe_embeddings(
    df=df,
    output_path=OUTPUT_PATH,
    text_column="Accident_Context_Text",
    batch_size=BATCH_SIZE,
)

print("\nLoading saved memmap...")

embeddings = np.memmap(
    OUTPUT_PATH,
    dtype=np.float32,
    mode="r",
    shape=(TEST_ROWS, 768),
)

print(
    "SAVED EMBEDDING SHAPE:",
    embeddings.shape,
)

print(
    "SAVED EMBEDDING TYPE:",
    embeddings.dtype,
)

print(
    "FIRST EMBEDDING NORM:",
    np.linalg.norm(
        embeddings[0]
    ),
)

print(
    "\nBERT PERSISTENCE TEST PASSED"
)