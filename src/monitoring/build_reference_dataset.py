import pandas as pd
from pathlib import Path
import joblib

# Paths
DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")
REF_DIR = Path("data/reference")

REF_DIR.mkdir(parents=True, exist_ok=True)

# 1️⃣ Load raw data
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges (Telco dataset specific)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 2️⃣ Feature engineering (IDENTIQUE au training)
df["customer_value"] = df["tenure"] * df["MonthlyCharges"]
df["high_monthly_charges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

# 3️⃣ Drop target + ID
TARGET_COL = "Churn"
ID_COL = "customerID"

X = df.drop(columns=[TARGET_COL, ID_COL])

# 4️⃣ Load fitted preprocessor
preprocessor = joblib.load(PREPROCESSOR_PATH)

# 5️⃣ Apply preprocessing
X_preprocessed = preprocessor.transform(X)

# 6️⃣ Rebuild DataFrame
feature_names = preprocessor.get_feature_names_out()
X_ref = pd.DataFrame(X_preprocessed, columns=feature_names)

# 7️⃣ Save reference dataset
X_ref.to_parquet(REF_DIR / "X_train_ref.parquet")

print("✅ Reference dataset created for Evidently")
print(f"Shape: {X_ref.shape}")
