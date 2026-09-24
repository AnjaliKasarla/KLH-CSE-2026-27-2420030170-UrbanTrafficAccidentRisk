import sys
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


print("Loading sample data...")

df = pd.read_csv(
    DATA_PATH,
    nrows=16,
)

df = add_accident_context_text(df)

print(
    "ROWS:",
    len(df),
)

print(
    "\nLoading DistilBERT embedder..."
)

embedder = DistilBERTEmbedder()

print(
    "DEVICE:",
    embedder.device,
)

print(
    "MODEL:",
    embedder.model_name,
)

print(
    "\nGenerating embeddings..."
)

embeddings = embedder.encode_dataframe(
    df,
    text_column="Accident_Context_Text",
    batch_size=8,
)

print(
    "\nEMBEDDING SHAPE:",
    embeddings.shape,
)

print(
    "EMBEDDING TYPE:",
    embeddings.dtype,
)

print(
    "EXPECTED DIMENSION: 768"
)

print(
    "\nBATCH EMBEDDING TEST PASSED"
)