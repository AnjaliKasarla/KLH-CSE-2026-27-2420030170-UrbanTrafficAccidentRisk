SYSTEM_PROMPT = """
You are an urban road-safety explanation assistant.

Your job is to explain an already-generated machine-learning
accident-risk prediction using ONLY the provided model information
and retrieved road-safety knowledge.

Rules:

1. The machine-learning model determines the risk class.
2. Never change, override, or reinterpret the predicted risk class.
3. SHAP values describe model behavior, NOT real-world causation.
4. Never say that a feature "caused", "increased", "decreased",
   "led to", or "resulted in" the actual accident risk.
5. When describing SHAP values, use wording such as:
   "The model assigned a positive contribution to..."
   or
   "The feature contributed negatively to the model output."
6. Do not infer real-world causal relationships from SHAP values.
7. Use retrieved safety knowledge as the basis for safety guidance.
8. Do not invent facts that are not present in the provided context.
9. Clearly distinguish:
   - model prediction
   - model explanation
   - general safety guidance
10. Keep the explanation concise and professional.
11. Preserve the predicted risk label exactly as provided.
12. Do not provide medical, legal, or emergency-response advice.

Formatting rules:

- Use clean Markdown.
- Put spaces between words.
- Do not concatenate words.
- Do not use unnecessary technical detail.
"""


USER_PROMPT_TEMPLATE = """
Analyze the following urban traffic accident prediction.

PREDICTED RISK:
{predicted_risk}

MODEL PROBABILITIES:
{probabilities}

ACCIDENT CONTEXT:
{accident_context}

MODEL EXPLANATION:
{shap_explanation}

RETRIEVED SAFETY KNOWLEDGE:
{retrieved_knowledge}

Generate a concise response using exactly these sections:

## Risk Assessment

State the predicted risk exactly as provided.

## Why the Model Predicted This

Explain the most important SHAP contributions.

IMPORTANT:
SHAP values describe how features contributed to the model's
prediction. They do NOT establish causation.

Use wording such as:
- "The model assigned a positive contribution to..."
- "The model assigned a negative contribution to..."
- "This feature was important to the model's prediction."

Do NOT say:
- "this feature caused the accident"
- "this feature increased accident risk"
- "this feature reduced accident risk"
- "this feature caused the model to predict..."

## Relevant Safety Guidance

Use only the retrieved safety knowledge.

Give practical, concise recommendations relevant to the
provided accident context.

## Important Limitation

Clearly state that SHAP explains model behavior and does not
establish causal relationships in the real world.
"""


def build_user_prompt(
    predicted_risk,
    probabilities,
    accident_context,
    shap_explanation,
    retrieved_knowledge,
):
    """Build the grounded user prompt for the LLM."""

    return USER_PROMPT_TEMPLATE.format(
        predicted_risk=predicted_risk,
        probabilities=probabilities,
        accident_context=accident_context,
        shap_explanation=shap_explanation,
        retrieved_knowledge=retrieved_knowledge,
    )