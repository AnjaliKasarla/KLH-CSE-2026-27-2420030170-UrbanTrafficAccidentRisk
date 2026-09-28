"""Automated tests for the road-safety RAG pipeline."""

from pathlib import Path

from src.rag.chunker import chunk_documents
from src.rag.document_loader import load_markdown_documents
from src.rag.embeddings import EmbeddingModel
from src.rag.retriever import SafetyRetriever
from src.rag.vector_store import VectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_BASE = (
    PROJECT_ROOT
    / "Code"
    / "backend"
    / "data"
    / "external"
    / "knowledge_base"
)

RAG_STORE = PROJECT_ROOT / "data" / "interim" / "rag"


def test_knowledge_base_exists():
    assert KNOWLEDGE_BASE.exists()
    assert KNOWLEDGE_BASE.is_dir()

    documents = list(KNOWLEDGE_BASE.glob("*.md"))

    assert len(documents) > 0


def test_knowledge_base_loader():
    documents = load_markdown_documents(
        str(KNOWLEDGE_BASE)
    )

    assert len(documents) > 0

    first = documents[0]

    assert "content" in first
    assert "source" in first
    assert first["source"].endswith(".md")
    assert len(first["content"].strip()) > 0


def test_document_chunking():
    documents = load_markdown_documents(
        str(KNOWLEDGE_BASE)
    )

    chunks = chunk_documents(documents)

    assert len(chunks) > 0

    first = chunks[0]

    assert "chunk_id" in first
    assert "source" in first
    assert "content" in first
    assert len(first["content"].strip()) > 0


def test_chunk_size_configuration():
    documents = load_markdown_documents(
        str(KNOWLEDGE_BASE)
    )

    chunks = chunk_documents(
        documents,
        chunk_size=1200,
        chunk_overlap=100,
    )

    assert len(chunks) > 0

    for chunk in chunks:
        assert len(chunk["content"]) <= 1200


def test_embedding_model():
    model = EmbeddingModel()

    embeddings = model.encode_documents(
        [
            "junction safety and traffic signals",
            "safe driving during adverse weather",
        ]
    )

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384


def test_vector_store_exists():
    assert RAG_STORE.exists()
    assert RAG_STORE.is_dir()

    embeddings_file = RAG_STORE / "embeddings.npy"
    metadata_file = RAG_STORE / "metadata.npy"

    assert embeddings_file.exists()
    assert metadata_file.exists()


def test_vector_store_loading():
    vector_store = VectorStore(
        storage_dir=RAG_STORE
    )

    assert vector_store.embeddings is not None
    assert vector_store.metadata is not None

    assert len(vector_store.embeddings) > 0
    assert len(vector_store.metadata) > 0

    assert vector_store.embeddings.shape[1] == 384


def test_rag_retrieval():
    retriever = SafetyRetriever(
        storage_dir=str(RAG_STORE)
    )

    results = retriever.retrieve(
        "How should drivers approach junctions safely?",
        top_k=3,
    )

    assert len(results) > 0
    assert len(results) <= 3

    first = results[0]

    assert "content" in first
    assert "source" in first
    assert "score" in first

    assert isinstance(first["content"], str)
    assert isinstance(first["source"], str)
    assert isinstance(first["score"], float)

    assert first["score"] > 0


def test_rag_weather_retrieval():
    retriever = SafetyRetriever(
        storage_dir=str(RAG_STORE)
    )

    results = retriever.retrieve(
        "What should drivers do during adverse weather?",
        top_k=3,
    )

    assert len(results) > 0

    contents = " ".join(
        result["content"].lower()
        for result in results
    )

    assert any(
        term in contents
        for term in (
            "weather",
            "speed",
            "visibility",
            "following distance",
        )
    )
