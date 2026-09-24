"""Generate grounded natural-language explanations from model and RAG outputs."""

import re
from typing import Dict, List, Optional

from .client import LLMClient
from .prompts import SYSTEM_PROMPT, build_user_prompt


class ResponseGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    @staticmethod
    def _clean_response(text: str) -> str:
        """Clean common formatting artifacts from generated Markdown."""

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Repair common word-concatenation artifacts.
        replacements = {
            "conditionsrequire": "conditions require",
            "thereal": "the real",
            "androad": "and road",
            "signalsand": "signals and",
            "indicatorswhen": "indicators when",
            "according toapplicable": "according to applicable",
            "toapplicable": "to applicable",
            "road-safety": "road-safety",
        }

        for incorrect, correct in replacements.items():
            text = text.replace(incorrect, correct)

        # Normalize excessive spaces while preserving Markdown structure.
        text = re.sub(r"[ \t]+", " ", text)

        # Remove trailing whitespace from every line.
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        return text.strip()

    def generate(
        self,
        predicted_risk: str,
        probabilities: Dict[str, float],
        accident_context: str,
        shap_explanation: str,
        retrieved_knowledge: List[Dict[str, object]],
    ) -> str:

        probability_text = ", ".join(
            f"{label}: {probability:.3f}"
            for label, probability in probabilities.items()
        )

        knowledge_text_parts = []

        for rank, item in enumerate(retrieved_knowledge, start=1):
            knowledge_text_parts.append(
                f"[Source {rank}] "
                f"{item['source']} "
                f"(similarity={item['score']:.4f})\n"
                f"{item['content']}"
            )

        retrieved_knowledge_text = "\n\n".join(knowledge_text_parts)

        user_prompt = build_user_prompt(
            predicted_risk=predicted_risk,
            probabilities=probability_text,
            accident_context=accident_context,
            shap_explanation=shap_explanation,
            retrieved_knowledge=retrieved_knowledge_text,
        )

        response = self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        return self._clean_response(response)