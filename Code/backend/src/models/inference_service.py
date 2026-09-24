"""
Production inference service for Urban Traffic Accident Risk.

Pipeline:
    API input
        -> feature engineering
        -> preprocessing
        -> XGBoost prediction
        -> local SHAP explanation
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.explainability.shap_explainer import (
    calculate_shap_values,
    create_explainer,
    get_local_feature_contributions,
)
from src.features.fusion import build_feature_dataset
from src.rag.rag_pipeline import RAGPipeline
from src.rag.context_builder import build_accident_context, build_rag_query
from src.llm.response_generator import ResponseGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "xgboost_final.joblib"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "xgboost_final_preprocessor.joblib"
)


CLASS_NAMES = [
    "Fatal",
    "Serious",
    "Slight",
]


MODEL_FEATURES = [
    "Junction_Control",
    "Junction_Detail",
    "Latitude",
    "Light_Conditions",
    "Local_Authority_(District)",
    "Carriageway_Hazards",
    "Longitude",
    "Police_Force",
    "Road_Surface_Conditions",
    "Road_Type",
    "Speed_limit",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
    "Vehicle_Type",
    "Accident_Day",
    "Accident_Month_Num",
    "Accident_Week",
    "Accident_Hour",
    "Time_Period",
    "Accident_Day_Of_Year",
    "Accident_Minute",
    "Is_Peak_Hour",
    "Hour_Sin",
    "Hour_Cos",
    "Weather_Category",
    "Road_Surface_Category",
    "Light_Category",
    "Speed_Limit_Category",
    "Junction_Type",
    "Road_Category",
    "Vehicle_Category",
    "Lat_Long_Interaction",
    "Area_Category",
    "Latitude_Region",
    "Longitude_Region",
    "Local_Authority_(District)_Group",
    "Police_Force_Group",
]


class InferenceService:
    """Load and serve the trained accident-risk model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
    ) -> None:

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found: {preprocessor_path}"
            )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(
            preprocessor_path
        )

        self.shap_explainer = create_explainer(
            self.model
        )

        self.rag_pipeline = RAGPipeline()
        self.response_generator = ResponseGenerator()
        self._validate_model_contract()

    def _validate_model_contract(self) -> None:
        """Validate compatibility with the trained model."""

        trained_features = list(
            self.preprocessor.feature_names_in_
        )

        if trained_features != MODEL_FEATURES:
            raise RuntimeError(
                "Inference feature contract mismatch.\n"
                f"Expected: {MODEL_FEATURES}\n"
                f"Found: {trained_features}"
            )

    @staticmethod
    def _get_time_period(
        hour: int | None,
    ) -> str:

        if hour is None:
            return "Unknown"

        if 5 <= hour < 12:
            return "Morning"

        if 12 <= hour < 17:
            return "Afternoon"

        if 17 <= hour < 21:
            return "Evening"

        if 21 <= hour <= 23:
            return "Night"

        return "Late_Night"

    @staticmethod
    def _build_input_dataframe(
        payload: dict,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            [
                {
                    "Accident Date": payload["accident_date"],
                    "Time": payload["time"],
                    "Junction_Control": payload["junction_control"],
                    "Junction_Detail": payload["junction_detail"],
                    "Latitude": payload["latitude"],
                    "Light_Conditions": payload["light_conditions"],
                    "Local_Authority_(District)": (
                        payload["local_authority_district"]
                    ),
                    "Carriageway_Hazards": (
                        payload["carriageway_hazards"]
                    ),
                    "Longitude": payload["longitude"],
                    "Police_Force": payload["police_force"],
                    "Road_Surface_Conditions": (
                        payload["road_surface_conditions"]
                    ),
                    "Road_Type": payload["road_type"],
                    "Speed_limit": payload["speed_limit"],
                    "Urban_or_Rural_Area": (
                        payload["urban_or_rural_area"]
                    ),
                    "Weather_Conditions": (
                        payload["weather_conditions"]
                    ),
                    "Vehicle_Type": payload["vehicle_type"],
                }
            ]
        )

    def build_features(
        self,
        payload: dict,
    ) -> pd.DataFrame:

        raw_df = self._build_input_dataframe(
            payload
        )

        engineered_df = build_feature_dataset(
            raw_df
        )

        parsed_time = pd.to_datetime(
            raw_df["Time"],
            format="%H:%M",
            errors="coerce",
        )

        hour = parsed_time.dt.hour.iloc[0]

        engineered_df["Time_Period"] = (
            self._get_time_period(
                None if pd.isna(hour)
                else int(hour)
            )
        )

        return engineered_df[
            MODEL_FEATURES
        ].copy()

    @staticmethod
    def _format_shap_feature_name(feature_name: str) -> str:
        """Convert encoded preprocessing names into readable labels."""

        if "__" in feature_name:
            feature_name = feature_name.split("__", 1)[1]

        if "_" in feature_name:
            base_features = [
                "Local_Authority_(District)",
                "Accident_Day_Of_Year",
                "Accident_Month_Num",
                "Accident_Week",
                "Accident_Hour",
                "Accident_Minute",
                "Speed_Limit_Category",
                "Urban_or_Rural_Area",
                "Junction_Detail",
                "Junction_Control",
                "Light_Conditions",
                "Road_Surface_Conditions",
                "Weather_Conditions",
                "Carriageway_Hazards",
                "Vehicle_Type",
                "Road_Type",
                "Police_Force",
                "Area_Category",
                "Road_Category",
                "Vehicle_Category",
            ]

            for base in sorted(base_features, key=len, reverse=True):
                if feature_name.startswith(base + "_"):
                    value = feature_name[len(base) + 1:]
                    return f"{base.replace('_', ' ')}: {value.replace('_', ' ')}"

        return feature_name.replace("_", " ")

    def _calculate_shap(
        self,
        transformed,
        predicted_class: int,
    ) -> list[dict]:

        shap_values = calculate_shap_values(
            self.shap_explainer,
            transformed,
        )

        contributions = (
            get_local_feature_contributions(
                shap_values=shap_values,
                feature_names=list(
                    self.preprocessor.get_feature_names_out()
                ),
                sample_index=0,
                class_index=predicted_class,
            )
        )

        top_contributions = (
            contributions.head(10)
        )

        return [
            {
                "feature": self._format_shap_feature_name(
                    str(row["feature"])
                ),
                "contribution": float(
                    row["shap_value"]
                ),
            }
            for _, row in top_contributions.iterrows()
        ]

    def predict(
        self,
        payload: dict,
    ) -> dict:

        features = self.build_features(
            payload
        )

        transformed = self.preprocessor.transform(
            features
        )

        prediction = int(
            self.model.predict(
                transformed
            )[0]
        )

        probabilities = self.model.predict_proba(
            transformed
        )[0]

        predicted_risk = CLASS_NAMES[prediction]

        shap_contributions = self._calculate_shap(
            transformed,
            prediction,
        )

        probability_dict = {
            "Fatal": float(probabilities[0]),
            "Serious": float(probabilities[1]),
            "Slight": float(probabilities[2]),
        }

        shap_for_rag = [
            {
                "feature": item["feature"],
                "value": item["contribution"],
                "direction": (
                    "positive"
                    if item["contribution"] > 0
                    else "negative"
                ),
            }
            for item in shap_contributions
        ]

        accident_features = {
            key: value
            for key, value in payload.items()
            if value is not None
        }

        accident_context = build_accident_context(
            accident_features=accident_features,
            predicted_risk=predicted_risk,
            probabilities=probability_dict,
            shap_contributions=shap_for_rag,
        )

        rag_query = build_rag_query(
            accident_features=accident_features,
            predicted_risk=predicted_risk,
        )

        retrieved_knowledge = self.rag_pipeline.retriever.retrieve(
            query=rag_query,
            top_k=3,
        )

        shap_explanation = "\n".join(
            f"- {item['feature']}: {item['contribution']:+.4f}"
            for item in shap_contributions
        )

        explanation = self.response_generator.generate(
            predicted_risk=predicted_risk,
            probabilities=probability_dict,
            accident_context=accident_context,
            shap_explanation=shap_explanation,
            retrieved_knowledge=retrieved_knowledge,
        )

        return {
            "predicted_risk": predicted_risk,
            "probabilities": {
                "fatal": float(probabilities[0]),
                "serious": float(probabilities[1]),
                "slight": float(probabilities[2]),
            },
            "shap_contributions": shap_contributions,
            "retrieved_knowledge": retrieved_knowledge,
            "accident_context": accident_context,
            "rag_query": rag_query,
            "explanation": explanation,
            "features": features,
            "transformed_shape": transformed.shape,
        }

