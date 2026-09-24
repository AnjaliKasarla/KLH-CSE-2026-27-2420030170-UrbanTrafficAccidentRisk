\# Urban Traffic Accident Risk Assessment \& Prediction — Backend



This directory contains the complete backend, machine-learning, NLP, explainability, RAG, and LLM pipeline for the \*\*Urban Traffic Accident Risk Assessment and Prediction\*\* system.



\---



\## 1. Backend Architecture



```text

&#x20;                   ┌──────────────────────────────┐

&#x20;                   │      Accident Input Data     │

&#x20;                   └──────────────┬───────────────┘

&#x20;                                  │

&#x20;                                  ▼

&#x20;                   ┌──────────────────────────────┐

&#x20;                   │ Data Collection \& Validation  │

&#x20;                   └──────────────┬───────────────┘

&#x20;                                  │

&#x20;                                  ▼

&#x20;                   ┌──────────────────────────────┐

&#x20;                   │ Preprocessing \& EDA           │

&#x20;                   └──────────────┬───────────────┘

&#x20;                                  │

&#x20;                   ┌──────────────┴──────────────┐

&#x20;                   ▼                             ▼

&#x20;         ┌──────────────────┐          ┌──────────────────┐

&#x20;         │ Structured       │          │ Accident Context │

&#x20;         │ Features         │          │ Text Generation  │

&#x20;         └────────┬─────────┘          └────────┬─────────┘

&#x20;                  │                             │

&#x20;                  │                    ┌────────┴─────────┐

&#x20;                  │                    ▼                  ▼

&#x20;                  │               TF-IDF             BERT /

&#x20;                  │                                  DistilBERT

&#x20;                  │                    │                  │

&#x20;                  └────────────────────┴──────────────────┘

&#x20;                                       │

&#x20;                                       ▼

&#x20;                             Feature Engineering

&#x20;                                       │

&#x20;                                       ▼

&#x20;                             Feature Fusion / Selection

&#x20;                                       │

&#x20;                                       ▼

&#x20;                        ┌────────────────────────────┐

&#x20;                        │ Model Training \& Evaluation │

&#x20;                        │ LR / DT / RF / SVM / XGBoost│

&#x20;                        └──────────────┬─────────────┘

&#x20;                                       │

&#x20;                                       ▼

&#x20;                              Risk Classification

&#x20;                                       │

&#x20;                                       ▼

&#x20;                             SHAP + LIME Explanation

&#x20;                                       │

&#x20;                                       ▼

&#x20;                             RAG Safety Retrieval

&#x20;                                       │

&#x20;                                       ▼

&#x20;                             Hugging Face LLM

&#x20;                                       │

&#x20;                                       ▼

&#x20;                                 FastAPI API

&#x20;                                       │

&#x20;                                       ▼

&#x20;                             React Dashboard

2\. Project Pipeline



The backend is organized into four major phases.



Phase 1 — Data Collection \& Preprocessing

Dataset loading

Data validation

Duplicate detection

Missing-value handling

Categorical encoding

Data preprocessing

Exploratory Data Analysis

Phase 2 — NLP \& Feature Engineering

Accident-context text generation

TF-IDF baseline

BERT / DistilBERT contextual embeddings

Temporal feature engineering

Weather features

Road features

Location features

Feature fusion

Feature selection



The project does not contain native free-text accident narratives in the primary Road Accident dataset. The NLP text is therefore derived from structured accident-context fields such as road type, weather, lighting, junction, vehicle type, and urban/rural context.



Phase 3 — Model Training \& Evaluation



The backend supports multiple machine-learning models:



Logistic Regression

Decision Tree

Random Forest

SVM

XGBoost



The final inference pipeline uses the trained XGBoost model.



Phase 4 — Explainability, RAG, LLM \& API

SHAP global/local explanations

LIME local explanations

Road-safety knowledge retrieval

Hugging Face LLM response generation

FastAPI inference API



The machine-learning model remains responsible for the risk prediction. RAG and the LLM provide contextual safety guidance and explanations rather than determining the risk class.



3\. Directory Structure

backend/

│

├── api/

│   ├── main.py

│   └── schemas.py

│

├── configs/

│   ├── config.yaml

│   └── model\_config.yaml

│

├── data/

│   ├── raw/

│   ├── interim/

│   ├── processed/

│   └── external/

│       └── knowledge\_base/

│           └── road\_safety\_guidelines.md

│

├── docs/

│

├── models/

│   ├── artifacts/

│   └── metadata/

│

├── notebooks/

│

├── reports/

│   ├── figures/

│   ├── metrics/

│   └── final/

│

├── scripts/

│

├── src/

│   ├── data/

│   ├── evaluation/

│   ├── explainability/

│   ├── features/

│   ├── llm/

│   ├── models/

│   ├── nlp/

│   ├── preprocessing/

│   ├── rag/

│   └── utils/

│

├── tests/

│

├── requirements.txt

├── requirements-dev.txt

└── .env.example



4\. Machine Learning

Target



The primary prediction target is:



Accident\_Severity



with the following classes:



Fatal

Serious

Slight

Final Model



The final model is an XGBoost multiclass classifier.



The trained inference pipeline consists of:



Input

&#x20; ↓

Feature Engineering

&#x20; ↓

Preprocessor

&#x20; ↓

XGBoost

&#x20; ↓

Class Probabilities

&#x20; ↓

Predicted Accident Severity

5\. NLP Pipeline



The primary dataset does not provide native accident-description text.



Therefore, the system creates a structured accident-context representation using available fields such as:



Junction control

Junction detail

Light conditions

Road surface

Road type

Urban/rural area

Weather

Vehicle type

Speed limit



This derived context is processed through:



&#x20;    Structured Accident Fields

&#x20;                ↓

&#x20;      Derived Accident Context

&#x20;                ↓

&#x20;        Text Preprocessing

&#x20;                ↓

&#x20;      ┌───────────────┐

&#x20;      │                    │

&#x20;      ▼                    ▼

&#x20;    TF-IDF              DistilBERT

&#x20;      │                    │

&#x20;      └───────┬───────┘

&#x20;                 ▼

&#x20;       Text Representations



TF-IDF is used as a baseline representation, while DistilBERT provides contextual embeddings.



6\. Explainability



The system uses both global and local explainability.



SHAP



SHAP is used to identify feature contributions to model predictions.



Prediction

&#x20;   ↓

&#x20; SHAP

&#x20;   ↓

Feature Contributions



SHAP values describe model behavior and should not be interpreted as causal relationships.



LIME



LIME is used for local explanations of individual predictions.



7\. RAG Pipeline



The RAG component retrieves relevant road-safety guidance from the project's knowledge base.



Safety Query / Risk Context

&#x20;         ↓

&#x20;      Embedding

&#x20;         ↓

&#x20;   Vector Similarity

&#x20;         ↓

&#x20;Relevant Knowledge

&#x20;         ↓

&#x20;Context Builder



The knowledge base is located at:



data/external/knowledge\_base/road\_safety\_guidelines.md



The vector store uses sentence-transformer embeddings and semantic similarity retrieval.

8\. LLM Pipeline



The LLM does not predict the accident risk.



Instead:



ML Prediction

&#x20;    +

SHAP Explanation

&#x20;    +

Retrieved Safety Knowledge

&#x20;    ↓

Hugging Face LLM

&#x20;    ↓

Grounded User Explanation



The generated response contains:



Risk Assessment

Why the Model Predicted This

Relevant Safety Guidance

Important Limitation



The LLM is instructed to preserve the ML model's predicted risk and use retrieved knowledge as supporting safety guidance.



9\. FastAPI



The API is located in:



api/main.py

Health Check

GET /health

Prediction

POST /predict



The prediction endpoint accepts accident context and returns:



Predicted risk

Class probabilities

SHAP contributions

Retrieved safety knowledge

LLM-generated explanation

10\. Environment Configuration



Create a local .env file based on:



.env.example



Example:



LLM\_PROVIDER=huggingface

HF\_MODEL=Qwen/Qwen3-4B-Instruct-2507

HF\_TOKEN=your\_huggingface\_token\_here



Never commit the real .env file or API tokens.



11\. Installation



From the repository root:



.\\.venv\\Scripts\\Activate.ps1



Install backend dependencies:



pip install -r Code\\backend\\requirements.txt

12\. Running the Backend



Move into the backend directory:



cd Code\\backend



Start FastAPI:



uvicorn api.main:app --reload



The API will be available locally through the FastAPI server.



13\. Reproducing Generated Artifacts



Large datasets and generated artifacts are intentionally excluded from Git version control.



They remain part of the project pipeline and can be regenerated.



Typical workflow:



Dataset

&#x20; ↓

Preprocessing

&#x20; ↓

Feature Engineering

&#x20; ↓

TF-IDF / BERT

&#x20; ↓

Model Training

&#x20; ↓

Evaluation

&#x20; ↓

SHAP / LIME

&#x20; ↓

RAG Vector Store

&#x20; ↓

FastAPI Inference



Relevant scripts are located in:



scripts/



including:



generate\_bert\_embeddings.py

run\_tfidf\_baseline.py

build\_features.py

train.py

evaluate.py

run\_shap.py

run\_lime.py

build\_knowledge\_base.py

run\_end\_to\_end.py

14\. Large Local Artifacts



The following generated resources are intentionally excluded from GitHub:



Raw accident datasets

BERT embedding matrices

TF-IDF matrices

Generated model binaries

RAG vector-store arrays

Generated reports



These files remain available in the local development environment and can be reproduced using the backend pipeline.



15\. Testing



Backend tests are located in:



tests/



Run:



pytest



For a backend runtime check:



python -c "from src.models.inference\_service import InferenceService; s=InferenceService(); print('BACKEND RUNTIME: OK')"

16\. Design Principle



The system follows a clear separation of responsibilities:



Machine Learning → predicts accident severity



SHAP / LIME → explains model behavior



RAG → retrieves relevant safety knowledge



LLM → generates grounded explanations and recommendations



FastAPI → exposes inference services



React Frontend → presents results to users



This separation prevents the generative AI components from replacing the trained accident-risk classifier.

