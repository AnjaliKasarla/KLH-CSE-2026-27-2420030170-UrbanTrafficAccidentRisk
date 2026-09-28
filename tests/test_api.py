"""Automated tests for the Urban Traffic Accident Risk API."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_PAYLOAD = {
    "accident_date": "2026-09-25",
    "time": "18:30",
    "junction_control": "Give way or uncontrolled",
    "junction_detail": "Not at junction",
    "light_conditions": "Daylight",
    "local_authority_district": "Unknown",
    "carriageway_hazards": "None",
    "police_force": "Unknown",
    "road_surface_conditions": "Dry",
    "road_type": "Single carriageway",
    "speed_limit": 50,
    "urban_or_rural_area": "Urban",
    "weather_conditions": "Fine no high winds",
    "vehicle_type": "Car",
    "latitude": 17.385,
    "longitude": 78.4867,
}


@pytest.fixture(scope="module")
def prediction_response():
    """Run one real prediction and reuse it across API tests."""

    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    return response


def test_health_endpoint():
    """The health endpoint should report an operational service."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "XGBoost"
    assert data["explainability"] == "SHAP"
    assert data["knowledge_retrieval"] == "RAG"
    assert data["language_model"] == "Hugging Face LLM"


def test_predict_endpoint(prediction_response):
    """A valid accident context should produce a complete prediction."""

    data = prediction_response.json()

    assert data["predicted_risk"] in {"Fatal", "Serious", "Slight"}

    assert "probabilities" in data

    assert set(data["probabilities"]) == {
        "fatal",
        "serious",
        "slight",
    }

    probabilities = data["probabilities"]

    assert all(0 <= value <= 1 for value in probabilities.values())

    assert abs(sum(probabilities.values()) - 1.0) < 1e-5


def test_predict_returns_explainability(prediction_response):
    """Prediction response should contain SHAP contributions."""

    data = prediction_response.json()

    assert "shap_contributions" in data
    assert isinstance(data["shap_contributions"], list)
    assert len(data["shap_contributions"]) > 0

    first = data["shap_contributions"][0]

    assert "feature" in first
    assert "contribution" in first


def test_predict_returns_rag_knowledge(prediction_response):
    """Prediction response should contain retrieved safety knowledge."""

    data = prediction_response.json()

    assert "retrieved_knowledge" in data
    assert isinstance(data["retrieved_knowledge"], list)
    assert len(data["retrieved_knowledge"]) > 0

    first = data["retrieved_knowledge"][0]

    assert "content" in first
    assert "source" in first
    assert "similarity" in first


def test_predict_returns_llm_explanation(prediction_response):
    """Prediction response should contain the grounded explanation."""

    data = prediction_response.json()

    assert "explanation" in data
    assert isinstance(data["explanation"], str)
    assert len(data["explanation"].strip()) > 0

    assert "Risk Assessment" in data["explanation"]
    assert "Why the Model Predicted This" in data["explanation"]
    assert "Relevant Safety Guidance" in data["explanation"]
    assert "Important Limitation" in data["explanation"]


def test_predict_rejects_invalid_payload():
    """Invalid requests should fail validation."""

    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["speed_limit"] = "invalid"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422