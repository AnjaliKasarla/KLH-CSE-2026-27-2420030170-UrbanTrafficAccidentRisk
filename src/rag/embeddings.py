from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate semantic embeddings for RAG documents and queries."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Generate embeddings for document chunks."""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a search query."""
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


if __name__ == "__main__":
    from document_loader import load_markdown_documents
    from chunker import chunk_documents

    knowledge_base_dir = "data/external/knowledge_base"

    documents = load_markdown_documents(knowledge_base_dir)
    chunks = chunk_documents(documents)

    texts = [chunk["content"] for chunk in chunks]

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.encode_documents(texts)

    print(f"\nChunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First vector norm: {np.linalg.norm(embeddings[0]):.4f}")