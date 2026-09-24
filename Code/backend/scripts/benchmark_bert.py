import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.nlp.text_features import (
    add_accident_context_text,
)

from src.nlp.bert_embeddings import (
    DistilBERTEmbedder,
)


DATA_PATH = (
    "data/raw/road_accident/"
    "Road Accident Data.csv"
)

SAMPLE_SIZE = 128
BATCH_SIZE = 8


print("Loading benchmark data...")

df = pd.read_csv(
    DATA_PATH,
    nrows=SAMPLE_SIZE,
)

df = add_accident_context_text(df)

print("ROWS:", len(df))
print("BATCH SIZE:", BATCH_SIZE)

print("\nLoading DistilBERT...")

embedder = DistilBERTEmbedder()

print("DEVICE:", embedder.device)

print("\nStarting benchmark...")

start_time = time.perf_counter()

embeddings = embedder.encode_dataframe(
    df,
    text_column="Accident_Context_Text",
    batch_size=BATCH_SIZE,
)

elapsed = time.perf_counter() - start_time

print("\n" + "=" * 50)
print("BERT BENCHMARK RESULTS")
print("=" * 50)

print(f"Records processed : {len(df)}")
print(f"Embedding shape   : {embeddings.shape}")
print(f"Elapsed time      : {elapsed:.2f} seconds")
print(
    f"Time per record   : "
    f"{elapsed / len(df):.4f} seconds"
)

estimated_seconds = (
    elapsed / len(df) * 307972
)

estimated_hours = (
    estimated_seconds / 3600
)

print(
    f"Estimated full dataset time : "
    f"{estimated_hours:.2f} hours"
)

print("=" * 50)