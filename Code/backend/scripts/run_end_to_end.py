"""
End-to-end Urban Traffic Accident Risk pipeline.

Flow:
    Engineered features
        ↓
    Fitted preprocessor
        ↓
    XGBoost prediction
        ↓
    SHAP local explanation
        ↓
    RAG safety retrieval
        ↓
    Hugging Face LLM
        ↓
    Grounded explanation
"""

import sys
from pathlib import Path


# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import joblib
import pandas as pd


from src.explainability.shap_explainer import (
    create_explainer,
    calculate_shap_values,
    get_local_feature_contributions,
    get_top_positive_contributions,
    get_top_negative_contributions,
)

from src.rag.rag_pipeline import RAGPipeline

from src.llm.response_generator import ResponseGenerator


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final"
    / "features.csv"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final"
    / "target.csv"
)

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

SAMPLE_INDEX = 0


# -------------------------------------------------------------------
# SHAP feature-name formatting
# -------------------------------------------------------------------

def format_shap_feature_name(
    feature_name: str,
) -> str:
    """
    Convert sklearn encoded feature names into
    readable feature names.

    Examples
    --------
    numerical__Speed_limit
        -> Speed Limit

    categorical__Road_Type_Single carriageway
        -> Road Type: Single carriageway

    categorical__Weather_Category_Clear
        -> Weather Category: Clear
    """

    # Remove sklearn transformer prefix.
    if "__" in feature_name:
        _, feature_name = (
            feature_name.split(
                "__",
                1,
            )
        )

    categorical_features = [
        "Junction_Control",
        "Junction_Detail",
        "Light_Conditions",
        "Local_Authority_(District)",
        "Carriageway_Hazards",
        "Police_Force",
        "Road_Surface_Conditions",
        "Road_Type",
        "Urban_or_Rural_Area",
        "Weather_Conditions",
        "Vehicle_Type",
        "Time_Period",
        "Weather_Category",
        "Road_Surface_Category",
        "Light_Category",
        "Speed_Limit_Category",
        "Junction_Type",
        "Road_Category",
        "Vehicle_Category",
        "Area_Category",
        "Local_Authority_(District)_Group",
        "Police_Force_Group",
    ]

    # Handle one-hot encoded categorical features.
    for feature in categorical_features:

        prefix = feature + "_"

        if feature_name.startswith(prefix):

            value = feature_name[
                len(prefix):
            ]

            readable_feature = (
                feature
                .replace("_", " ")
            )

            return (
                f"{readable_feature}: "
                f"{value}"
            )

    # Handle numerical features.
    return (
        feature_name
        .replace("_", " ")
        .strip()
    )


# -------------------------------------------------------------------
# Load artifacts
# -------------------------------------------------------------------

def load_artifacts():
    """
    Load trained model, fitted preprocessor,
    engineered features, and target.
    """

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    features = pd.read_csv(
        FEATURES_PATH
    )

    target = pd.read_csv(
        TARGET_PATH
    ).squeeze("columns")

    return (
        model,
        preprocessor,
        features,
        target,
    )


# -------------------------------------------------------------------
# Build accident context
# -------------------------------------------------------------------

def build_accident_features(
    row: pd.Series,
) -> dict:
    """
    Convert one engineered feature row
    into a compact accident context for RAG.
    """

    preferred_features = [
        "Speed_limit",
        "Weather_Conditions",
        "Road_Surface_Conditions",
        "Light_Conditions",
        "Road_Type",
        "Urban_or_Rural_Area",
        "Junction_Detail",
        "Vehicle_Type",
        "Junction_Control",
        "Carriageway_Hazards",
        "Latitude",
        "Longitude",
        "Accident_Hour",
        "Time_Period",
        "Is_Peak_Hour",
    ]

    return {
        feature: row[feature]
        for feature in preferred_features
        if feature in row.index
    }


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def main():

    print("=" * 80)
    print(
        "URBAN TRAFFIC ACCIDENT RISK — END-TO-END PIPELINE"
    )
    print("=" * 80)

    # ---------------------------------------------------------------
    # 1. Load artifacts
    # ---------------------------------------------------------------

    (
        model,
        preprocessor,
        features,
        target,
    ) = load_artifacts()

    print("\n[1] Artifacts loaded")

    print(
        f"Features: {features.shape}"
    )

    print(
        f"Target: {target.shape}"
    )

    # ---------------------------------------------------------------
    # 2. Select sample
    # ---------------------------------------------------------------

    sample = features.iloc[
        [SAMPLE_INDEX]
    ]

    actual_risk = str(
        target.iloc[
            SAMPLE_INDEX
        ]
    )

    print("\n[2] Selected sample")

    print(
        f"Sample index: {SAMPLE_INDEX}"
    )

    print(
        f"Actual risk: {actual_risk}"
    )

    # ---------------------------------------------------------------
    # 3. Preprocess
    # ---------------------------------------------------------------

    X_processed = (
        preprocessor.transform(
            sample
        )
    )

    print("\n[3] Preprocessing")

    print(
        f"Encoded shape: "
        f"{X_processed.shape}"
    )

    # ---------------------------------------------------------------
    # 4. XGBoost prediction
    # ---------------------------------------------------------------

    prediction_encoded = int(
        model.predict(
            X_processed
        )[0]
    )

    probabilities_array = (
        model.predict_proba(
            X_processed
        )[0]
    )

    predicted_risk = (
        CLASS_NAMES[
            prediction_encoded
        ]
    )

    probabilities = {
        CLASS_NAMES[index]: float(
            probabilities_array[index]
        )
        for index in range(
            len(CLASS_NAMES)
        )
    }

    print(
        "\n[4] XGBoost prediction"
    )

    print(
        f"Predicted risk: "
        f"{predicted_risk}"
    )

    for (
        label,
        probability,
    ) in probabilities.items():

        print(
            f"{label}: "
            f"{probability:.4f}"
        )

    # ---------------------------------------------------------------
    # 5. SHAP explanation
    # ---------------------------------------------------------------

    print(
        "\n[5] SHAP explanation"
    )

    explainer = create_explainer(
        model
    )

    shap_values = (
        calculate_shap_values(
            explainer,
            X_processed,
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    contributions = (
        get_local_feature_contributions(
            shap_values=shap_values,
            feature_names=feature_names,
            sample_index=0,
            class_index=prediction_encoded,
        )
    )

    positive = (
        get_top_positive_contributions(
            contributions,
            top_n=5,
        )
    )

    negative = (
        get_top_negative_contributions(
            contributions,
            top_n=5,
        )
    )

    shap_lines = []

    # ---------------------------------------------------------------
    # Positive SHAP contributions
    # ---------------------------------------------------------------

    for _, row in (
        positive.iterrows()
    ):

        readable_name = (
            format_shap_feature_name(
                row["feature"]
            )
        )

        shap_lines.append(
            f"- {readable_name}: "
            f"{row['shap_value']:.4f} "
            f"(positive contribution)"
        )

    # ---------------------------------------------------------------
    # Negative SHAP contributions
    # ---------------------------------------------------------------

    for _, row in (
        negative.iterrows()
    ):

        readable_name = (
            format_shap_feature_name(
                row["feature"]
            )
        )

        shap_lines.append(
            f"- {readable_name}: "
            f"{row['shap_value']:.4f} "
            f"(negative contribution)"
        )

    shap_explanation = (
        "\n".join(
            shap_lines
        )
    )

    print(
        shap_explanation
    )

    # ---------------------------------------------------------------
    # 6. RAG retrieval
    # ---------------------------------------------------------------

    print(
        "\n[6] RAG retrieval"
    )

    accident_features = (
        build_accident_features(
            sample.iloc[0]
        )
    )

    rag_pipeline = (
        RAGPipeline()
    )

    rag_result = (
        rag_pipeline.run(
            accident_features=(
                accident_features
            ),
            predicted_risk=(
                predicted_risk
            ),
            probabilities=(
                probabilities
            ),
            shap_contributions=[
                {
                    "feature": (
                        format_shap_feature_name(
                            row["feature"]
                        )
                    ),
                    "value": float(
                        row["shap_value"]
                    ),
                    "direction": (
                        "positive"
                        if row["shap_value"] > 0
                        else "negative"
                    ),
                }
                for _, row in (
                    contributions.head(
                        10
                    ).iterrows()
                )
            ],
            top_k=3,
        )
    )

    retrieved_knowledge = (
        rag_result[
            "retrieved_knowledge"
        ]
    )

    for (
        rank,
        item,
    ) in enumerate(
        retrieved_knowledge,
        start=1,
    ):

        print(
            f"Rank {rank}: "
            f"{item['source']} "
            f"(score="
            f"{item['score']:.4f})"
        )

    # ---------------------------------------------------------------
    # 7. Hugging Face LLM
    # ---------------------------------------------------------------

    print(
        "\n[7] Hugging Face LLM"
    )

    response_generator = (
        ResponseGenerator()
    )

    final_response = (
        response_generator.generate(
            predicted_risk=(
                predicted_risk
            ),
            probabilities=(
                probabilities
            ),
            accident_context=(
                rag_result[
                    "accident_context"
                ]
            ),
            shap_explanation=(
                shap_explanation
            ),
            retrieved_knowledge=(
                retrieved_knowledge
            ),
        )
    )

    # ---------------------------------------------------------------
    # 8. Final output
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL GROUNDED EXPLANATION"
    )

    print(
        "=" * 80
    )

    print(
        final_response
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PIPELINE COMPLETE"
    )

    print(
        "=" * 80
    )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()