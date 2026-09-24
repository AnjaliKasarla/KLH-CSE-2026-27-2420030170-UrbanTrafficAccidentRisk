# System Architecture

## High-Level Flow

Data Collection
→ Preprocessing
→ EDA
→ NLP/Text Processing
→ Feature Engineering
→ Feature Selection
→ Model Training
→ Model Evaluation
→ Best Model Selection
→ Risk Prediction
→ Risk Classification
→ Explainability
→ RAG
→ LLM
→ Dashboard
→ Testing & Deployment

## NLP Architecture

Accident Text
→ Text Cleaning
→ TF-IDF Baseline
→ BERT/DistilBERT Contextual Embeddings
→ Text Features
→ Feature Fusion with Structured Features

## AI Responsibility Separation

- ML models: risk prediction/classification.
- SHAP/LIME: model explanation.
- RAG: retrieval of relevant safety knowledge.
- LLM: grounded explanation, recommendations and user-facing assistance.
