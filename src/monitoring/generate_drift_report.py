import pandas as pd
from pathlib import Path
import mlflow

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ======================================================
# Paths
# ======================================================
REFERENCE_PATH = Path("data/reference/X_train_ref.parquet")
PROD_DIR = Path("data/prod_batches")
REPORT_DIR = Path("reports/evidently")

PROD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# MLflow config — FORCE LOCAL FILE STORE
# ======================================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("telco-churn-prod")

# ======================================================
# 1️⃣ Load reference dataset
# ======================================================
ref = pd.read_parquet(REFERENCE_PATH)

# ======================================================
# 2️⃣ Create a simulated production batch
# (later: this will come from /predict_csv)
# ======================================================
current = ref.sample(300, random_state=42)

batch_path = PROD_DIR / "batch_example.parquet"
current.to_parquet(batch_path)

print(f"✅ Production batch saved: {batch_path}")

# ======================================================
# 3️⃣ Build Evidently Data Drift report
# ======================================================
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=ref,
    current_data=current
)

# ======================================================
# 4️⃣ Save HTML report
# ======================================================
report_path = REPORT_DIR / "data_drift_report.html"
report.save(report_path)

print(f"✅ Evidently drift report generated: {report_path}")

# ======================================================
# 5️⃣ Log artifacts to MLflow
# ======================================================
with mlflow.start_run(run_name="evidently-data-drift"):
    mlflow.log_artifact(str(report_path), artifact_path="evidently")
    mlflow.log_artifact(str(batch_path), artifact_path="evidently_batches")

print("✅ Evidently report and batch logged to MLflow")
