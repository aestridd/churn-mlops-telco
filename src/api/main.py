import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

try:
    import mlflow  # optional at runtime
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore

from .schemas import PredictRequest, PredictResponse

app = FastAPI(title="Telco Churn Scoring API", version="1.0")

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "production_pipeline.joblib"

# =========================
# Env-driven config
# =========================
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
BUSINESS_THRESHOLD = float(os.getenv("BUSINESS_THRESHOLD", "0.40"))

MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "50000"))  # safety limit

SAVE_PROD_BATCHES = os.getenv("SAVE_PROD_BATCHES", "true").lower() == "true"
PROD_BATCH_DIR = Path(os.getenv("PROD_BATCH_DIR", str(PROJECT_ROOT / "data" / "production")))
PROD_BATCH_DIR.mkdir(parents=True, exist_ok=True)

# MLflow (optional)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn-prod")

pipeline: Optional[Any] = None


def _risk_level(p: float) -> str:
    # Business labels for UI: High / Medium / Low
    if p >= 0.70:
        return "High"
    if p >= BUSINESS_THRESHOLD:
        return "Medium"
    return "Low"


def _mlflow_enabled() -> bool:
    return (mlflow is not None) and bool(MLFLOW_TRACKING_URI)


@app.on_event("startup")
def load_assets() -> None:
    global pipeline

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    pipeline = joblib.load(MODEL_PATH)

    if _mlflow_enabled():
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model_path": str(MODEL_PATH),
        "threshold": BUSINESS_THRESHOLD,
        "max_batch_rows": MAX_BATCH_ROWS,
        "save_prod_batches": SAVE_PROD_BATCHES,
        "prod_batch_dir": str(PROD_BATCH_DIR),
        "mlflow_enabled": _mlflow_enabled(),
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME if _mlflow_enabled() else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not hasattr(pipeline, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model does not support predict_proba")

    df = pd.DataFrame([req.features])

    try:
        proba = float(pipeline.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    pred = int(proba >= BUSINESS_THRESHOLD)

    return PredictResponse(
        churn_probability=proba,
        churn_pred=pred,
        threshold=BUSINESS_THRESHOLD,
    )


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Batch scoring endpoint.

    Returns:
      - summary: KPI metrics computed over the full uploaded batch
      - customers: all scored customers, sorted by customerID
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not hasattr(pipeline, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model does not support predict_proba")

    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    t0 = time.perf_counter()

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    batch_size_raw = int(len(df))
    if batch_size_raw == 0:
        raise HTTPException(status_code=400, detail="Empty CSV")

    if batch_size_raw > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Batch too large: {batch_size_raw} rows. "
                f"Max allowed: {MAX_BATCH_ROWS}. Please split the file."
            ),
        )

    saved_batch_path: Optional[Path] = None
    if SAVE_PROD_BATCHES:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = Path(file.filename or "upload.csv").stem
        saved_batch_path = PROD_BATCH_DIR / f"{safe_name}_batch_{ts}_{batch_size_raw}.csv"
        try:
            df.to_csv(saved_batch_path, index=False)
        except Exception:
            saved_batch_path = None

    # Separate ID column if present
    id_candidates = ["customerID", "client_id", "id_client"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    X = df.drop(columns=[id_col]) if id_col else df

    try:
        proba = pipeline.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # Build business result
    result = df.copy()
    result["churn_probability"] = proba
    result["churn_pred"] = (result["churn_probability"] >= BUSINESS_THRESHOLD).astype(int)
    result["risk_level"] = result["churn_probability"].apply(_risk_level)

    # Monitoring metrics
    batch_size = int(len(result))
    avg_proba = float(result["churn_probability"].mean())
    p95_proba = float(result["churn_probability"].quantile(0.95))
    churn_rate_pred = float(result["churn_pred"].mean())

    latency_ms = (time.perf_counter() - t0) * 1000.0
    rows_per_sec = batch_size / (latency_ms / 1000.0 + 1e-9)

    if _mlflow_enabled():
        try:
            with mlflow.start_run(run_name="prod_batch_scoring"):
                mlflow.log_metric("batch_size", batch_size)
                mlflow.log_metric("avg_churn_proba", avg_proba)
                mlflow.log_metric("p95_churn_proba", p95_proba)
                mlflow.log_metric("predicted_churn_rate", churn_rate_pred)
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("rows_per_sec", rows_per_sec)

                mlflow.log_param("threshold", BUSINESS_THRESHOLD)
                mlflow.log_param("model_path", str(MODEL_PATH))
                mlflow.log_param("save_prod_batches", SAVE_PROD_BATCHES)
                mlflow.log_param("max_batch_rows", MAX_BATCH_ROWS)

                if saved_batch_path is not None:
                    mlflow.log_param("saved_batch_path", str(saved_batch_path))
        except Exception:
            pass

    # Output: business columns only
    cols = [c for c in [id_col, "churn_probability", "churn_pred", "risk_level"] if c is not None]
    out = result[cols].copy()

    if id_col and id_col != "customerID":
        out = out.rename(columns={id_col: "customerID"})

    # JSON-safe
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)

    # Sort by ID by default
    if "customerID" in out.columns:
        out["customerID"] = out["customerID"].astype(str)
        out = out.sort_values("customerID", ascending=True)

    # Summary over full batch
    risk = out["risk_level"].astype(str)
    payload = {
        "summary": {
            "n_scored": int(len(out)),
            "high": int(risk.str.contains("High").sum()),
            "medium": int(risk.str.contains("Medium").sum()),
            "low": int(risk.str.contains("Low").sum()),
            "avg_proba": float(pd.to_numeric(out["churn_probability"], errors="coerce").mean()),
        },
        "customers": out.to_dict(orient="records"),
    }

    return JSONResponse(payload)
