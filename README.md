TELCO CHURN PREDICTOR
END-TO-END MLOPS PROJECT

==================================================

PROJECT OVERVIEW

This project demonstrates a production-ready machine learning application for predicting customer churn in the telecom industry.

It showcases a complete MLOps pipeline:

Model training

API deployment

UI integration

Docker containerization

Cloud deployment with auto-redeploy

The project is designed as a portfolio-grade example of real-world ML deployment practices.

LIVE DEMO

API (FastAPI + Swagger):
https://telco-churn-predictor-ibeh.onrender.com/docs

Health Check:
https://telco-churn-predictor-ibeh.onrender.com/health

UI (Streamlit):
https://churn-web.onrender.com/

OBJECTIVE

Customer churn is a major business issue in telecom.

The goal of this project is to:

Predict churn probability for customers

Support batch predictions via CSV upload

Provide a simple interface for non-technical users

Demonstrate robust and reproducible ML deployment

MACHINE LEARNING MODEL

Task: Binary classification (Churn / No Churn)

Model:

Logistic Regression (scikit-learn)

Pipeline components:

SimpleImputer

StandardScaler

OneHotEncoder

ColumnTransformer

Model serialization:

joblib

Production model file:
models/production_pipeline.joblib

Important note:
Special care was taken to align scikit-learn versions between training and inference, a common real-world MLOps issue.

ARCHITECTURE

Streamlit UI --> FastAPI API --> ML Pipeline

UI and API are deployed as separate services

Communication via HTTP

Fully containerized with Docker

DOCKER & DEPLOYMENT

Docker:

Dockerfile.api for FastAPI service

Dockerfile.web for Streamlit UI

.dockerignore used to keep images lightweight

Production model explicitly included in the image

Deployment:

Hosted on Render

Docker-based web services

Auto-deploy enabled on main branch

Health check endpoint configured (/health)

API ENDPOINTS

GET /health
Returns service status.
Response:
{ "status": "ok" }

POST /predict_csv

Upload a CSV file

Returns churn predictions and probabilities

Accepts multipart/form-data

Testable directly via Swagger UI

STREAMLIT UI

The Streamlit application allows users to:

Upload a CSV file

Send it to the API

Display churn predictions interactively

The API endpoint is configured via environment variable:
API_URL=https://telco-churn-predictor-ibeh.onrender.com

TECH STACK

Python 3.11

scikit-learn 1.3.0

FastAPI

Streamlit

Docker

Render

joblib

pandas

numpy

LOCAL DEVELOPMENT

Run API locally:
docker build -f Dockerfile.api -t churn-api .
docker run -p 8000:8000 -e PORT=8000 churn-api

API available at:
http://localhost:8000/docs

Run UI locally:
docker build -f Dockerfile.web -t churn-web .
docker run -p 8501:8501 -e PORT=8501 -e API_URL=http://localhost:8000
 churn-web

UI available at:
http://localhost:8501

MLOPS LESSONS HIGHLIGHTED

Version mismatches between training and inference can break production

Pinning ML dependencies is critical

Separating API and UI improves robustness

Health checks are essential for cloud deployment

Auto-deploy enables simple CI/CD for ML services

PROJECT STATUS

Model trained: YES
API deployed: YES
UI deployed: YES
End-to-end pipeline functional: YES

==================================================