from typing import Dict, List

from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class SafetyRetriever:
    """
    Retrieve relevant road-safety knowledge
    using semantic similarity search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        storage_dir: str = "data/interim/rag",
    ):
        self.embedding_model = EmbeddingModel(
            model_name=model_name
        )

        self.vector_store = VectorStore(
            storage_dir=storage_dir
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, object]]:
        """
        Retrieve the most relevant safety-knowledge chunks.
        """

        query_embedding = (
            self.embedding_model.encode_query(
                query
            )
        )

        # Retrieve extra candidates first so low-value
        # document metadata chunks can be filtered out.
        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=min(top_k + 3, len(self.vector_store.metadata)),
        )

        excluded_chunks = {
            "road_safety_guidelines.md::chunk_0",
            "road_safety_guidelines.md::chunk_1",
            "road_safety_guidelines.md::chunk_11",
        }

        results = [
            result
            for result in candidates
            if result["chunk_id"] not in excluded_chunks
        ]

        return results[:top_k]


if __name__ == "__main__":

    retriever = SafetyRetriever()

    query = (
        "Road safety guidance for "
        "wet roads and high speed"
    )

    results = retriever.retrieve(
        query=query,
        top_k=3,
    )

    print("\nSEMANTIC SEARCH RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(f"\nRank: {rank}")
        print(
            f"Score: {result['score']:.4f}"
        )
        print(
            f"Source: {result['source']}"
        )
        print(
            f"Chunk: {result['chunk_id']}"
        )
        print(
            f"\n{result['content'][:500]}"
        )