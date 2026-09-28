"""Tests for the NLP and text-representation pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.nlp.bert_embeddings import (
    EMBEDDING_DIMENSION,
    DistilBERTEmbedder,
)
from src.nlp.text_features import (
    TEXT_FIELDS,
    add_accident_context_text,
    create_accident_context_text,
)
from src.nlp.tfidf import TfidfEncoder


def test_accident_context_text():
    row = pd.Series(
        {
            "Junction_Control": "Give way or uncontrolled",
            "Junction_Detail": "Not at junction",
            "Light_Conditions": "Daylight",
            "Road_Surface_Conditions": "Dry",
            "Road_Type": "Single carriageway",
            "Urban_or_Rural_Area": "Urban",
            "Weather_Conditions": "Fine no high winds",
            "Vehicle_Type": "Car",
            "Speed_limit": 50,
        }
    )

    text = create_accident_context_text(row)

    assert isinstance(text, str)
    assert len(text) > 0

    assert "Junction Control: give way or uncontrolled" in text
    assert "Junction Detail: not at junction" in text
    assert "Light Conditions: daylight" in text
    assert "Road Surface Conditions: dry" in text
    assert "Road Type: single carriageway" in text
    assert "Urban or Rural Area: urban" in text
    assert "Weather Conditions: fine no high winds" in text
    assert "Vehicle Type: car" in text
    assert "Speed limit: 50" in text


def test_accident_context_handles_missing_values():
    row = pd.Series(
        {
            "Weather_Conditions": None,
            "Road_Type": np.nan,
            "Speed_limit": 50,
        }
    )

    text = create_accident_context_text(row)

    assert "Weather Conditions: unknown" in text
    assert "Road Type: unknown" in text
    assert "Speed limit: 50.0" in text


def test_accident_context_uses_available_fields():
    row = pd.Series(
        {
            "Weather_Conditions": "Rain",
            "Road_Type": "Single carriageway",
        }
    )

    text = create_accident_context_text(row)

    assert "Weather Conditions: rain" in text
    assert "Road Type: single carriageway" in text

    assert "Junction Control:" not in text
    assert "Vehicle Type:" not in text


def test_text_fields_configuration():
    assert len(TEXT_FIELDS) == 9

    expected_fields = {
        "Junction_Control",
        "Junction_Detail",
        "Light_Conditions",
        "Road_Surface_Conditions",
        "Road_Type",
        "Urban_or_Rural_Area",
        "Weather_Conditions",
        "Vehicle_Type",
        "Speed_limit",
    }

    assert set(TEXT_FIELDS) == expected_fields


def test_add_accident_context_text():
    df = pd.DataFrame(
        {
            "Weather_Conditions": [
                "Fine no high winds",
                "Raining no high winds",
            ],
            "Road_Type": [
                "Single carriageway",
                "Dual carriageway",
            ],
            "Vehicle_Type": [
                "Car",
                "Motorcycle",
            ],
        }
    )

    result = add_accident_context_text(df)

    assert "Accident_Context_Text" in result.columns
    assert len(result) == 2

    assert isinstance(
        result.loc[0, "Accident_Context_Text"],
        str,
    )

    assert (
        "Weather Conditions: fine no high winds"
        in result.loc[0, "Accident_Context_Text"]
    )


def test_tfidf_encoder():
    texts = [
        "dry road daylight car",
        "wet road rain motorcycle",
        "dry road daylight vehicle",
    ]

    encoder = TfidfEncoder(
        max_features=100,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
    )

    matrix = encoder.fit_transform(texts)

    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0
    assert matrix.nnz > 0

    feature_names = encoder.get_feature_names()

    assert len(feature_names) == matrix.shape[1]


def test_tfidf_transform_after_fit():
    train_texts = [
        "dry road daylight car",
        "wet road rain motorcycle",
        "clear weather road",
    ]

    encoder = TfidfEncoder(
        max_features=100,
        min_df=1,
        max_df=1.0,
    )

    encoder.fit_transform(train_texts)

    transformed = encoder.transform(
        ["dry road car"]
    )

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(
        encoder.get_feature_names()
    )


def test_tfidf_save_and_load(tmp_path):
    texts = [
        "dry road daylight car",
        "wet road rain motorcycle",
        "clear weather road",
    ]

    encoder = TfidfEncoder(
        max_features=100,
        min_df=1,
        max_df=1.0,
    )

    original_matrix = encoder.fit_transform(
        texts
    )

    model_path = tmp_path / "tfidf.joblib"

    encoder.save(model_path)

    assert model_path.exists()
    assert model_path.stat().st_size > 0

    loaded = TfidfEncoder.load(model_path)

    loaded_matrix = loaded.transform(texts)

    assert loaded_matrix.shape == original_matrix.shape

    assert np.allclose(
        loaded_matrix.toarray(),
        original_matrix.toarray(),
    )


def test_distilbert_empty_batch():
    embedder = DistilBERTEmbedder(
        device="cpu"
    )

    embeddings = embedder.encode_batch([])

    assert embeddings.shape == (
        0,
        EMBEDDING_DIMENSION,
    )

    assert embeddings.dtype == np.float32


def test_distilbert_batch_embeddings():
    embedder = DistilBERTEmbedder(
        device="cpu"
    )

    embeddings = embedder.encode_batch(
        [
            "dry road daylight car",
            "wet road rain motorcycle",
        ]
    )

    assert embeddings.shape == (
        2,
        EMBEDDING_DIMENSION,
    )

    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()


def test_distilbert_dataframe_embeddings():
    embedder = DistilBERTEmbedder(
        device="cpu"
    )

    df = pd.DataFrame(
        {
            "Accident_Context_Text": [
                "dry road daylight car",
                "wet road rain motorcycle",
            ]
        }
    )

    embeddings = embedder.encode_dataframe(
        df,
        batch_size=2,
    )

    assert embeddings.shape == (
        2,
        EMBEDDING_DIMENSION,
    )

    assert embeddings.dtype == np.float32


def test_distilbert_missing_text_column():
    embedder = DistilBERTEmbedder(
        device="cpu"
    )

    df = pd.DataFrame(
        {
            "wrong_column": [
                "test text"
            ]
        }
    )

    try:
        embedder.encode_dataframe(df)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Accident_Context_Text" in str(exc)
