import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv(override=True)


class LLMClient:
    """Provider-independent LLM client using Hugging Face Inference Providers."""

    def __init__(self, provider=None, model=None):
        self.provider = provider or os.getenv(
            "LLM_PROVIDER",
            "huggingface",
        )

        self.model = model or os.getenv(
            "HF_MODEL",
            "Qwen/Qwen3-4B-Instruct-2507",
        )

        if self.provider != "huggingface":
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}"
            )

        self.token = os.getenv("HF_TOKEN")

        if not self.token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not configured."
            )

        self.client = InferenceClient(
            provider="auto",
            api_key=self.token,
        )

    def generate(self, system_prompt, user_prompt):
        """Generate a response using Hugging Face Inference Providers."""

        response = self.client.chat_completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_tokens=800,
            temperature=0.2,
        )

        return response.choices[0].message.content