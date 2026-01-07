📊 Telco Churn — End-to-End MLOps Pipeline
🎯 Project Overview

This project implements a production-ready MLOps pipeline for customer churn prediction, with a strong focus on business impact, automation, and usability by non-technical teams.
The goal is not only to train a churn model, but to cover the entire machine learning lifecycle, from data preparation to deployment, monitoring, and automated retraining.

🧠 Business Problem

In highly competitive B2B and B2C markets, retaining high-value customers is critical.
However, commercial teams often lack actionable prioritization tools to decide which customers to focus on first.

This project provides:
- a churn prediction model
- a REST API for scoring customers
- a business-oriented UI to prioritize retention actions

🧱 Project Architecture

The pipeline covers the full MLOps lifecycle:
- Data preparation & feature engineering
- Model training & evaluation
- Experiment tracking & versioning
- REST API deployment
- Business-oriented web UI
- Monitoring & data drift detection
- Automated retraining
- Model promotion & rollback strategy

📁 Repository Structure
churn-mlops-telco/
├── src/
│   ├── api/                 # FastAPI inference service
│   ├── app_web.py           # Streamlit business UI
│   ├── retraining/          # Automated retraining logic
│   └── monitoring/          # Drift detection (Evidently)
│
├── models/
│   └── production_pipeline.joblib   # Single production model
│
├── data/
│   └── new/                 # Placeholder for new incoming data
│
├── notebooks/
│   ├── eda/
│   ├── ml/
│   ├── mlflow/
│   └── test_api/
│
├── .github/workflows/
│   └── retrain.yml          # GitHub Actions retraining workflow
│
├── requirements.txt
├── README.md
└── run_project.ps1

🤖 Modeling

Task: Binary classification (churn / non-churn)
Models tested:
Logistic Regression
Random Forest
Gradient Boosting
Final model: Logistic Regression
class_weight="balanced"
Optimized for recall on churn class
Business threshold: 0.40
The final pipeline includes:
preprocessing (ColumnTransformer)
feature engineering
model inference

All serialized into a single pipeline artifact.

📈 Experiment Tracking & Versioning

MLflow is used for:
- experiment tracking
- metrics logging
- artifact storage
- retraining traceability

Experiments:

- telco-churn-final — training
- telco-churn-retraining — automated retraining
- telco-churn-prod — production monitoring

Only one production model is versioned in the repository:

models/production_pipeline.joblib

🚀 API — FastAPI
Endpoints

GET /health
Health check & configuration overview

POST /predict
Predict churn for a single customer (JSON)

POST /predict_csv
Batch scoring via CSV upload

Features

Stateless & scalable API

Business threshold applied server-side

Batch size safety limit

Latency & throughput monitoring

Optional MLflow logging

🖥️ Business UI — Streamlit

A non-technical, commercial-friendly interface:

CSV upload

Batch scoring via API

Client prioritization

Risk classification:

🔴 High
🟠 Medium
🟢 Low

KPI counters

Filtering (high-risk only)

Export only selected customers

📊 Monitoring & Drift Detection

Evidently used for data drift detection
- Reference dataset built from training data
- Production batches optionally stored
- Drift reports generated as HTML
- Drift signals used as retraining triggers

🔁 Automated Retraining

Retraining is automated using GitHub Actions (lightweight, no infrastructure overhead).

Triggers:

New data detected in data/new/

Data drift detected

Significant metric degradation

Manual trigger (workflow_dispatch)

Workflow:

Retrain model

Log metrics to MLflow

Save candidate model

Promote or rollback based on business rules

▶️ Quickstart
1. Install dependencies
pip install -r requirements.txt

2. Start the API
uvicorn src.api.main:app --reload

3. Start the UI
streamlit run src/app_web.py

⚙️ Environment Variables (optional)
API_URL=http://localhost:8000
BUSINESS_THRESHOLD=0.40
MAX_BATCH_ROWS=50000
MLFLOW_TRACKING_URI=file:///path/to/mlruns

✅ Key Design Choices

Simplicity over over-engineering

No Kubernetes

No Docker (by design)

GitHub Actions instead of Airflow

Single production model

Clear separation between training, inference, and business usage

📌 Final Notes

This project is designed as:

- a realistic production-grade MLOps example
- a portfolio-ready project
- a foundation adaptable to real company data

✨ Status: Production-ready
✨ Focus: Business value + MLOps best practices