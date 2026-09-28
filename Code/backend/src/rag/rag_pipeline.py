from typing import Dict, List, Optional

from .context_builder import (
    build_accident_context,
    build_rag_query,
)
from .retriever import SafetyRetriever


class RAGPipeline:
    """
    End-to-end RAG pipeline for accident safety guidance.

    ML prediction remains authoritative for risk classification.
    RAG only retrieves relevant safety knowledge.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        storage_dir: str = "data/interim/rag",
    ):
        self.retriever = SafetyRetriever(
            model_name=model_name,
            storage_dir=storage_dir,
        )

    def run(
        self,
        accident_features: Dict[str, object],
        predicted_risk: str,
        probabilities: Optional[Dict[str, float]] = None,
        shap_contributions: Optional[
            List[Dict[str, object]]
        ] = None,
        top_k: int = 3,
    ) -> Dict[str, object]:
        """
        Build accident context, retrieve relevant safety knowledge,
        and return the complete grounded context.
        """

        # -----------------------------------------------------------
        # Build accident context
        # -----------------------------------------------------------

        accident_context = build_accident_context(
            accident_features=accident_features,
            predicted_risk=predicted_risk,
            probabilities=probabilities,
            shap_contributions=shap_contributions,
        )

        # -----------------------------------------------------------
        # Build semantic-search query
        # -----------------------------------------------------------

        rag_query = build_rag_query(
            accident_features=accident_features,
            predicted_risk=predicted_risk,
        )

        # -----------------------------------------------------------
        # Retrieve relevant safety knowledge
        # -----------------------------------------------------------

        retrieved_knowledge = self.retriever.retrieve(
            query=rag_query,
            top_k=top_k,
        )

        # -----------------------------------------------------------
        # Return complete RAG result
        # -----------------------------------------------------------

        return {
            "predicted_risk": predicted_risk,
            "probabilities": probabilities,
            "accident_context": accident_context,
            "rag_query": rag_query,
            "retrieved_knowledge": retrieved_knowledge,
        }


# -------------------------------------------------------------------
# Standalone RAG test
# -------------------------------------------------------------------

if __name__ == "__main__":

    pipeline = RAGPipeline()

    accident_features = {
        "Speed_limit": 60,
        "Weather_Conditions": "Rain",
        "Road_Surface_Conditions": "Wet",
        "Light_Conditions": "Darkness - lights lit",
        "Road_Type": "Single carriageway",
        "Urban_or_Rural_Area": "Urban",
        "Junction_Detail": "Crossroads",
    }

    probabilities = {
        "Fatal": 0.12,
        "Serious": 0.61,
        "Slight": 0.27,
    }

    shap_contributions = [
        {
            "feature": "Speed_limit",
            "value": 0.13,
            "direction": "positive",
        },
        {
            "feature": "Weather_Conditions",
            "value": 0.08,
            "direction": "positive",
        },
        {
            "feature": "Urban_or_Rural_Area",
            "value": -0.04,
            "direction": "negative",
        },
    ]

    result = pipeline.run(
        accident_features=accident_features,
        predicted_risk="Serious",
        probabilities=probabilities,
        shap_contributions=shap_contributions,
        top_k=3,
    )

    print("\nPREDICTED RISK")
    print("=" * 70)
    print(result["predicted_risk"])

    print("\nRAG QUERY")
    print("=" * 70)
    print(result["rag_query"])

    print("\nACCIDENT CONTEXT")
    print("=" * 70)
    print(result["accident_context"])

    print("\nRETRIEVED SAFETY KNOWLEDGE")
    print("=" * 70)

    for rank, item in enumerate(
        result["retrieved_knowledge"],
        start=1,
    ):
        print(f"\nRank: {rank}")
        print(f"Score: {item['score']:.4f}")
        print(f"Source: {item['source']}")
        print(f"Chunk: {item['chunk_id']}")
        print(f"\n{item['content'][:500]}")