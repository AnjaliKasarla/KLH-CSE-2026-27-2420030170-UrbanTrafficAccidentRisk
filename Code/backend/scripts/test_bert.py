import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

from src.nlp.text_features import add_accident_context_text


DATA_PATH = (
    "data/raw/road_accident/"
    "Road Accident Data.csv"
)

MODEL_NAME = "distilbert-base-uncased"


# --------------------------------------------------
# 1. Load only a small sample
# --------------------------------------------------

print("Loading sample data...")

df = pd.read_csv(
    DATA_PATH,
    nrows=3,
)

print("ROWS LOADED:", len(df))


# --------------------------------------------------
# 2. Create derived accident-context text
# --------------------------------------------------

df = add_accident_context_text(df)

texts = df[
    "Accident_Context_Text"
].tolist()

print(
    "NUMBER OF TEST TEXTS:",
    len(texts)
)

print("\nSAMPLE TEXT:")
print(texts[0])


# --------------------------------------------------
# 3. Load DistilBERT
# --------------------------------------------------

print("\nLoading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.eval()


# --------------------------------------------------
# 4. Tokenize
# --------------------------------------------------

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

print(
    "\nINPUT SHAPE:",
    inputs["input_ids"].shape
)


# --------------------------------------------------
# 5. Generate embeddings
# --------------------------------------------------

with torch.no_grad():

    outputs = model(**inputs)

    embeddings = (
        outputs.last_hidden_state[:, 0, :]
    )


# --------------------------------------------------
# 6. Verify embeddings
# --------------------------------------------------

print(
    "EMBEDDING SHAPE:",
    embeddings.shape
)

print(
    "EMBEDDING TYPE:",
    embeddings.dtype
)

print(
    "MODEL DEVICE:",
    next(model.parameters()).device
)

print(
    "\nBERT TEST COMPLETED SUCCESSFULLY"
)