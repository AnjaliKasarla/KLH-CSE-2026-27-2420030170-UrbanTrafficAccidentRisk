"""
FastAPI application for Urban Traffic Accident Risk Prediction.
"""

from fastapi import FastAPI, HTTPException

from api.schemas import (
    AccidentPredictionRequest,
    AccidentPredictionResponse,
    RiskProbabilities,
    SHAPContribution,
    RetrievedKnowledge,
)
from src.models.inference_service import InferenceService


app = FastAPI(
    title="Urban Traffic Accident Risk API",
    description=(
        "Inference API for urban traffic accident "
        "risk classification using XGBoost, SHAP, "
        "RAG, and an LLM-based explanation layer."
    ),
    version="1.0.0",
)


# Load inference service once when the API starts.
inference_service = InferenceService()


@app.get("/health")
def health_check() -> dict:
    """Return API health status."""

    return {
        "status": "healthy",
        "model": "XGBoost",
        "explainability": "SHAP",
        "knowledge_retrieval": "RAG",
        "language_model": "Hugging Face LLM",
        "service": "urban-traffic-accident-risk",
    }


@app.post(
    "/predict",
    response_model=AccidentPredictionResponse,
)
def predict(
    request: AccidentPredictionRequest,
) -> AccidentPredictionResponse:
    """
    Predict accident severity and return:

    - XGBoost risk prediction
    - class probabilities
    - SHAP feature contributions
    - retrieved road-safety knowledge
    - grounded LLM explanation
    """

    try:
        # Convert validated Pydantic request into a dictionary.
        result = inference_service.predict(
            request.model_dump()
        )

        # Convert SHAP results into API response objects.
        shap_contributions = [
            SHAPContribution(
                feature=item["feature"],
                contribution=float(item["contribution"]),
            )
            for item in result["shap_contributions"]
        ]

        # Convert RAG retrieval results into API response objects.
        retrieved_knowledge = [
            RetrievedKnowledge(
                content=item["content"],
                source=item["source"],
                similarity=float(item["score"]),
            )
            for item in result["retrieved_knowledge"]
        ]

        # Return the complete prediction response.
        return AccidentPredictionResponse(
            predicted_risk=result["predicted_risk"],
            probabilities=RiskProbabilities(
                **result["probabilities"]
            ),
            shap_contributions=shap_contributions,
            retrieved_knowledge=retrieved_knowledge,

            # Return the actual grounded LLM explanation.
            explanation=result["explanation"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc