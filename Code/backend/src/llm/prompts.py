SYSTEM_PROMPT = """
You are an urban road-safety explanation assistant.

Explain an already-generated machine-learning accident-risk prediction
using ONLY the supplied model output and retrieved safety knowledge.

STRICT RULES:

1. The ML model determines the risk class.
2. Never change the predicted risk.
3. SHAP describes model behavior, not causation.
4. Never claim that a feature caused an accident or caused real-world risk.
5. Never reinterpret a SHAP contribution.
6. Do not infer what a negative contribution means beyond:
   "The model assigned a negative contribution to [feature]."
7. Do not infer what a positive contribution means beyond:
   "The model assigned a positive contribution to [feature]."
8. Use only the supplied retrieved safety knowledge for recommendations.
9. Do not invent facts.
10. Preserve proper spaces between every word.
11. Do not merge words.
12. Do not repeat recommendations.
13. Keep the response concise and professional.

IMPORTANT SHAP LANGUAGE:

Use exactly these patterns:

"The model assigned a positive contribution to [feature]."

"The model assigned a negative contribution to [feature]."

Do NOT add interpretations such as:
- "reduced confidence"
- "increased confidence"
- "lower-risk scenario"
- "higher-risk scenario"
- "played a role"
- "caused"
- "increased risk"
- "decreased risk"

The SHAP section must describe model behavior only.

Return clean Markdown.
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

Generate EXACTLY these four sections:

## Risk Assessment

Write exactly one sentence:

"The predicted risk is [PREDICTED RISK]."

Preserve the supplied risk label exactly.

## Why the Model Predicted This

Use at most 4 bullet points.

For each selected SHAP feature, use ONLY one of these forms:

- "The model assigned a positive contribution to [feature]."
- "The model assigned a negative contribution to [feature]."

Do not explain, reinterpret, or infer anything beyond the supplied
SHAP contribution.

Do not add causal language.

## Relevant Safety Guidance

Use ONLY the retrieved safety knowledge.

Select up to 4 relevant recommendations.

Copy the meaning faithfully and ensure every word has proper spacing.

Do not invent recommendations.
Do not repeat the same recommendation.

## Important Limitation

Write exactly:

"SHAP explains how features contributed to the model's prediction; it does not establish causal relationships in the real world."

FINAL CHECK:

- Exactly four sections.
- Correct predicted risk.
- Maximum 4 SHAP bullets.
- Maximum 4 safety bullets.
- No causal claims.
- No SHAP reinterpretation.
- No invented facts.
- No concatenated words.
- No extra sections.
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
