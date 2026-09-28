from typing import Dict, List, Optional


def build_accident_context(
    accident_features: Dict[str, object],
    predicted_risk: str,
    probabilities: Optional[Dict[str, float]] = None,
    shap_contributions: Optional[List[Dict[str, object]]] = None,
) -> str:
    """
    Build a structured accident context for RAG retrieval.

    Important:
    - The ML model determines the risk class.
    - SHAP explains model behavior.
    - RAG provides external safety knowledge.
    - This function does not modify the prediction.
    """

    context_parts = []

    # ---------------------------------------------------------
    # 1. Model prediction
    # ---------------------------------------------------------
    context_parts.append(
        f"Predicted accident risk: {predicted_risk}."
    )

    # ---------------------------------------------------------
    # 2. Prediction probabilities
    # ---------------------------------------------------------
    if probabilities:
        probability_text = ", ".join(
            f"{label}: {probability:.3f}"
            for label, probability in probabilities.items()
        )

        context_parts.append(
            f"Model probabilities: {probability_text}."
        )

    # ---------------------------------------------------------
    # 3. Accident characteristics
    # ---------------------------------------------------------
    feature_text = []

    for feature, value in accident_features.items():

        if value is None:
            continue

        feature_name = feature.replace("_", " ")

        feature_text.append(
            f"{feature_name}: {value}"
        )

    if feature_text:
        context_parts.append(
            "Accident characteristics: "
            + "; ".join(feature_text)
            + "."
        )

    # ---------------------------------------------------------
    # 4. SHAP explanation
    # ---------------------------------------------------------
    if shap_contributions:

        shap_text = []

        for contribution in shap_contributions:

            feature = contribution.get("feature")
            value = contribution.get("value")
            direction = contribution.get("direction")

            if feature is None:
                continue

            feature_name = feature.replace("_", " ")

            if direction:
                shap_text.append(
                    f"{feature_name} ({direction}, "
                    f"contribution={value})"
                )
            else:
                shap_text.append(
                    f"{feature_name} "
                    f"(contribution={value})"
                )

        if shap_text:
            context_parts.append(
                "Important model explanation features: "
                + "; ".join(shap_text)
                + "."
            )

    return "\n".join(context_parts)


def build_rag_query(
    accident_features: Dict[str, object],
    predicted_risk: str,
) -> str:
    """
    Build a concise semantic-search query.

    The query focuses on environmental and road-safety
    characteristics rather than model-internal details.
    """

    query_parts = [
        "Road safety guidance",
        f"predicted accident risk {predicted_risk}",
    ]

    for feature, value in accident_features.items():

        if value is None:
            continue

        feature_name = feature.replace("_", " ")

        query_parts.append(
            f"{feature_name}: {value}"
        )

    return "; ".join(query_parts)


if __name__ == "__main__":

    accident_features = {
        "Speed_limit": 60,
        "Weather_Conditions": "Rain",
        "Road_Surface_Conditions": "Wet",
        "Light_Conditions": "Darkness - lights lit",
        "Road_Type": "Single carriageway",
        "Urban_or_Rural_Area": "Urban",
        "Junction_Detail": "Crossroads",
    }

    probabilities = {
        "Fatal": 0.12,
        "Serious": 0.61,
        "Slight": 0.27,
    }

    shap_contributions = [
        {
            "feature": "Speed_limit",
            "value": 0.13,
            "direction": "positive",
        },
        {
            "feature": "Weather_Conditions",
            "value": 0.08,
            "direction": "positive",
        },
        {
            "feature": "Urban_or_Rural_Area",
            "value": -0.04,
            "direction": "negative",
        },
    ]

    context = build_accident_context(
        accident_features=accident_features,
        predicted_risk="Serious",
        probabilities=probabilities,
        shap_contributions=shap_contributions,
    )

    query = build_rag_query(
        accident_features=accident_features,
        predicted_risk="Serious",
    )

    print("\nACCIDENT CONTEXT")
    print("=" * 70)
    print(context)

    print("\nRAG QUERY")
    print("=" * 70)
    print(query)