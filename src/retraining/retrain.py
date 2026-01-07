from __future__ import annotations

from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

THRESHOLD_PATH = MODELS_DIR / "threshold.json"
DEFAULT_THRESHOLD = 0.40


def load_threshold() -> float:
    if THRESHOLD_PATH.exists():
        return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    return DEFAULT_THRESHOLD


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def main():
    # --- Load data (retraining uses the same processed sets for demo) ---
    X_train = joblib.load(DATA_DIR / "X_train.joblib")
    X_test = joblib.load(DATA_DIR / "X_test.joblib")
    y_train = joblib.load(DATA_DIR / "y_train.joblib")
    y_test = joblib.load(DATA_DIR / "y_test.joblib")

    threshold = load_threshold()

    # --- Build pipeline (preprocessor + model) ---
    preprocessor = build_preprocessor(X_train)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # --- Train ---
    pipeline.fit(X_train, y_train)

    # --- Evaluate ---
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    recall_churn = recall_score(y_test, y_pred, pos_label=1)
    precision_churn = precision_score(y_test, y_pred, pos_label=1)
    f1_churn = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    # --- MLflow logging ---
    mlflow.set_tracking_uri(f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}")
    mlflow.set_experiment("telco-churn-retraining")

    with mlflow.start_run(run_name="retrain_logreg_demo"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("recall_churn", float(recall_churn))
        mlflow.log_metric("precision_churn", float(precision_churn))
        mlflow.log_metric("f1_churn", float(f1_churn))

        # Log text artifacts
        artifacts_dir = PROJECT_ROOT / "artifacts" / "retraining"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        (artifacts_dir / "classification_report.txt").write_text(report)
        (artifacts_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist(), indent=2))

        mlflow.log_artifact(str(artifacts_dir / "classification_report.txt"))
        mlflow.log_artifact(str(artifacts_dir / "confusion_matrix.json"))

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    print("✅ Retraining run logged to MLflow")
    print(f"ROC-AUC: {roc_auc:.4f} | Recall churn: {recall_churn:.4f}")

    # --- Optional: promote model (show-off but clean) ---
    # Save retrained model as a candidate (does not overwrite prod automatically)
    candidate_path = MODELS_DIR / "candidate_pipeline.joblib"
    joblib.dump(pipeline, candidate_path)
    print("✅ Candidate model saved:", candidate_path)


if __name__ == "__main__":
    main()
