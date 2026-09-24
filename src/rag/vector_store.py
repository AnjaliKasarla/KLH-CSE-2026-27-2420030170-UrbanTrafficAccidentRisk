from pathlib import Path
from typing import Dict, List

import numpy as np


class VectorStore:
    """
    Simple local vector store for RAG.

    Stores:
    - normalized embedding vectors
    - corresponding chunk metadata

    Cosine similarity is implemented as a dot product because
    embeddings are normalized.
    """

    def __init__(self, storage_dir: str = "data/interim/rag"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_path = self.storage_dir / "embeddings.npy"
        self.metadata_path = self.storage_dir / "metadata.npy"

        self.embeddings = None
        self.metadata = None

    def build(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, str]],
    ) -> None:
        """Build and persist the vector store."""

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.metadata = np.array(chunks, dtype=object)

        np.save(self.embeddings_path, self.embeddings)
        np.save(self.metadata_path, self.metadata, allow_pickle=True)

    def load(self) -> None:
        """Load a previously built vector store."""

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {self.embeddings_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {self.metadata_path}"
            )

        self.embeddings = np.load(self.embeddings_path)
        self.metadata = np.load(
            self.metadata_path,
            allow_pickle=True,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict[str, object]]:
        """
        Retrieve the most semantically similar chunks.

        Returns chunk metadata together with similarity scores.
        """

        if self.embeddings is None or self.metadata is None:
            raise RuntimeError(
                "Vector store is not loaded or built."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0:
            raise ValueError("Query embedding cannot be a zero vector.")

        query_embedding = query_embedding / query_norm

        similarities = self.embeddings @ query_embedding

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:
            chunk = dict(self.metadata[index])

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "score": float(similarities[index]),
                }
            )

        return results


if __name__ == "__main__":
    from chunker import chunk_documents
    from document_loader import load_markdown_documents
    from embeddings import EmbeddingModel

    knowledge_base_dir = "data/external/knowledge_base"

    # 1. Load documents
    documents = load_markdown_documents(knowledge_base_dir)

    # 2. Create chunks
    chunks = chunk_documents(documents)

    # 3. Generate embeddings
    embedding_model = EmbeddingModel()

    texts = [chunk["content"] for chunk in chunks]

    embeddings = embedding_model.encode_documents(texts)

    # 4. Build vector store
    vector_store = VectorStore()

    vector_store.build(
        embeddings=embeddings,
        chunks=chunks,
    )

    print(f"Chunks stored: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Storage directory: {vector_store.storage_dir}")

    # 5. Test semantic retrieval
    query = (
        "Accident occurred during heavy rain "
        "with high vehicle speed"
    )

    query_embedding = embedding_model.encode_query(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    print("\nSemantic Search Results")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Content:\n{result['content'][:400]}")