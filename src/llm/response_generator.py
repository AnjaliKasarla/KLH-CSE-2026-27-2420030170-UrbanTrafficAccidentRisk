"""Implementation for response_generator. Added during the corresponding project phase."""

from typing import Dict, List, Optional

from .client import LLMClient
from .prompts import SYSTEM_PROMPT, build_user_prompt

class ResponseGenerator:
    """
    Generate a grounded natural-language explanation
    from ML prediction, SHAP information, and RAG knowledge.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
    ):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        predicted_risk: str,
        probabilities: Dict[str, float],
        accident_context: str,
        shap_explanation: str,
        retrieved_knowledge: List[Dict[str, object]],
    ) -> str:
        """
        Generate a grounded explanation.

        The predicted risk is passed directly from the ML model
        and is never modified by this component.
        """

        probability_text = ", ".join(
            f"{label}: {probability:.3f}"
            for label, probability in probabilities.items()
        )

        knowledge_text_parts = []

        for rank, item in enumerate(
            retrieved_knowledge,
            start=1,
        ):
            knowledge_text_parts.append(
                f"[Source {rank}] "
                f"{item['source']} "
                f"(similarity={item['score']:.4f})\n"
                f"{item['content']}"
            )

        retrieved_knowledge_text = "\n\n".join(
            knowledge_text_parts
        )

        user_prompt = build_user_prompt(
            predicted_risk=predicted_risk,
            probabilities=probability_text,
            accident_context=accident_context,
            shap_explanation=shap_explanation,
            retrieved_knowledge=retrieved_knowledge_text,
        )

        return self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )


if __name__ == "__main__":
    print("ResponseGenerator initialized.")
    print(
        "Ready to generate grounded LLM explanations "
        "when an LLM API key is configured."
    )