"""Tests for the accident-risk model layer."""

import joblib

from src.models.inference_service import (
    CLASS_NAMES,
    MODEL_FEATURES,
    InferenceService,
    MODEL_PATH,
    PREPROCESSOR_PATH,
)
from src.models.xgboost_model import create_xgboost


def test_xgboost_factory():
    model = create_xgboost()
    params = model.get_params()

    assert model.__class__.__name__ == "XGBClassifier"
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 8
    assert params["learning_rate"] == 0.05
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8
    assert params["objective"] == "multi:softprob"
    assert params["num_class"] == 3
    assert params["eval_metric"] == "mlogloss"
    assert params["tree_method"] == "hist"
    assert params["random_state"] == 42


def test_model_artifacts_exist():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.is_file()
    assert MODEL_PATH.stat().st_size > 0

    assert PREPROCESSOR_PATH.exists()
    assert PREPROCESSOR_PATH.is_file()
    assert PREPROCESSOR_PATH.stat().st_size > 0


def test_model_artifact_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert model.__class__.__name__ == "XGBClassifier"


def test_preprocessor_artifact_can_be_loaded():
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")
    assert hasattr(preprocessor, "feature_names_in_")


def test_model_feature_contract():
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    trained_features = list(
        preprocessor.feature_names_in_
    )

    assert trained_features == MODEL_FEATURES
    assert len(MODEL_FEATURES) == 37


def test_class_configuration():
    assert CLASS_NAMES == [
        "Fatal",
        "Serious",
        "Slight",
    ]

    assert len(CLASS_NAMES) == 3


def test_inference_service_initialization():
    service = InferenceService()

    assert service.model is not None
    assert service.preprocessor is not None
    assert service.shap_explainer is not None
    assert service.rag_pipeline is not None
    assert service.response_generator is not None


def test_inference_service_feature_builder():
    service = InferenceService()

    payload = {
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

    features = service.build_features(payload)

    assert list(features.columns) == MODEL_FEATURES
    assert features.shape[0] == 1
    assert features.shape[1] == len(MODEL_FEATURES)
    assert features.loc[0, "Accident_Hour"] == 18
    assert features.loc[0, "Time_Period"] == "Evening"
