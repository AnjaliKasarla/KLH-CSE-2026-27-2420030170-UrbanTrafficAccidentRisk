"""
FastAPI application for Urban Traffic Accident Risk Prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
# Allows the React/Vite frontend to communicate with
# the FastAPI backend during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Inference service
# ---------------------------------------------------------
# Load the trained XGBoost model, preprocessor,
# SHAP, RAG and LLM components once.
inference_service = InferenceService()


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
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

        # Return complete prediction response.
        return AccidentPredictionResponse(
            predicted_risk=result["predicted_risk"],
            probabilities=RiskProbabilities(
                **result["probabilities"]
            ),
            shap_contributions=shap_contributions,
            retrieved_knowledge=retrieved_knowledge,
            explanation=result["explanation"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc