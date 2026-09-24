from typing import List
from pydantic import BaseModel, Field


class AccidentPredictionRequest(BaseModel):
    """Input payload for urban traffic accident risk prediction."""

    accident_date: str = Field(..., description="Accident date in YYYY-MM-DD format")
    time: str = Field(..., description="Accident time in HH:MM format")

    junction_control: str
    junction_detail: str
    light_conditions: str
    local_authority_district: str
    carriageway_hazards: str
    police_force: str
    road_surface_conditions: str
    road_type: str
    speed_limit: float
    urban_or_rural_area: str
    weather_conditions: str
    vehicle_type: str

    latitude: float
    longitude: float



class RiskProbabilities(BaseModel):
    fatal: float
    serious: float
    slight: float


class SHAPContribution(BaseModel):
    feature: str
    contribution: float


class RetrievedKnowledge(BaseModel):
    content: str
    source: str
    similarity: float


class AccidentPredictionResponse(BaseModel):
    predicted_risk: str
    probabilities: RiskProbabilities
    shap_contributions: List[SHAPContribution]
    retrieved_knowledge: List[RetrievedKnowledge]
    explanation: str
