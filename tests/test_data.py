"""Tests for the current project's data artifacts."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAG_STORE = DATA_ROOT / "interim" / "rag"


def test_data_directory_exists():
    assert DATA_ROOT.exists()
    assert DATA_ROOT.is_dir()


def test_rag_data_directory_exists():
    assert RAG_STORE.exists()
    assert RAG_STORE.is_dir()


def test_rag_embeddings_artifact_exists():
    embeddings_file = RAG_STORE / "embeddings.npy"

    assert embeddings_file.exists()
    assert embeddings_file.is_file()
    assert embeddings_file.stat().st_size > 0


def test_rag_metadata_artifact_exists():
    metadata_file = RAG_STORE / "metadata.npy"

    assert metadata_file.exists()
    assert metadata_file.is_file()
    assert metadata_file.stat().st_size > 0


def test_rag_embeddings_shape():
    embeddings = np.load(
        RAG_STORE / "embeddings.npy"
    )

    assert embeddings.ndim == 2
    assert embeddings.shape[0] > 0
    assert embeddings.shape[1] == 384


def test_rag_metadata_shape():
    metadata = np.load(
        RAG_STORE / "metadata.npy",
        allow_pickle=True,
    )

    assert metadata.ndim == 1
    assert len(metadata) > 0


def test_rag_embeddings_and_metadata_match():
    embeddings = np.load(
        RAG_STORE / "embeddings.npy"
    )

    metadata = np.load(
        RAG_STORE / "metadata.npy",
        allow_pickle=True,
    )

    assert embeddings.shape[0] == len(metadata)


def test_rag_metadata_contains_required_fields():
    metadata = np.load(
        RAG_STORE / "metadata.npy",
        allow_pickle=True,
    )

    first = metadata[0]

    assert isinstance(first, dict)
    assert "chunk_id" in first
    assert "source" in first
    assert "content" in first

    assert isinstance(first["chunk_id"], str)
    assert isinstance(first["source"], str)
    assert isinstance(first["content"], str)
