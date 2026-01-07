import os
import requests
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(page_title="Telco Churn – Client Prioritization", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
BUSINESS_THRESHOLD = float(os.getenv("BUSINESS_THRESHOLD", "0.40"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

ID_COLUMNS = ["customerID", "client_id", "id_client"]


# =========================
# Utils
# =========================
def ensure_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use risk_level returned by the API if present.
    If missing, compute it locally (fallback).
    """
    df = df.copy()

    if "churn_probability" in df.columns:
        df["churn_probability"] = pd.to_numeric(df["churn_probability"], errors="coerce")

    if "risk_level" in df.columns and df["risk_level"].notna().any():
        return df

    # Fallback (shouldn't happen if API is aligned)
    def _risk(p: float) -> str:
        if pd.isna(p):
            return "🟢 Low"
        if p >= 0.70:
            return "🔴 High"
        if p >= BUSINESS_THRESHOLD:
            return "🟠 Medium"
        return "🟢 Low"

    df["risk_level"] = df["churn_probability"].apply(_risk)
    return df


def build_business_view(df_scored: pd.DataFrame) -> pd.DataFrame:
    df = df_scored.copy()

    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)

    cols = [c for c in [id_col, "churn_probability", "risk_level"] if c is not None]
    df_view = df[cols].copy()

    # Normalize ID column name for business users
    if id_col and id_col != "customerID":
        df_view = df_view.rename(columns={id_col: "customerID"})

    df_view["churn_probability"] = pd.to_numeric(df_view["churn_probability"], errors="coerce").round(4)

    # Sort by risk then probability
    order = {"🔴 High": 0, "🟠 Medium": 1, "🟢 Low": 2, "High": 0, "Medium": 1, "Low": 2}
    df_view["_order"] = df_view["risk_level"].map(order).fillna(99)
    df_view = (
        df_view.sort_values(["_order", "churn_probability"], ascending=[True, False])
        .drop(columns="_order")
    )

    return df_view


# =========================
# UI
# =========================
st.title("📊 Telco Churn — Client Prioritization Tool")
st.write(
    "Upload a CSV file with client data to identify **high-risk customers** "
    "and prioritize retention actions."
)

with st.expander("⚙️ Settings", expanded=False):
    st.caption("These values can be overridden via environment variables.")
    st.write(
        {
            "API_URL": API_URL,
            "BUSINESS_THRESHOLD": BUSINESS_THRESHOLD,
            "REQUEST_TIMEOUT": REQUEST_TIMEOUT,
        }
    )

uploaded_file = st.file_uploader("📂 Upload client CSV", type=["csv"])

if not uploaded_file:
    st.info("⬆️ Upload a CSV file to start scoring clients.")
    st.stop()

# Optional preview before scoring
with st.expander("👀 Preview uploaded data", expanded=False):
    try:
        preview_df = pd.read_csv(uploaded_file, nrows=10)
        st.dataframe(preview_df, use_container_width=True)
    except Exception:
        st.caption("Preview unavailable (file could not be parsed as CSV).")

# Reset pointer because we read it once for preview
uploaded_file.seek(0)

with st.spinner("Scoring clients..."):
    try:
        response = requests.post(
            f"{API_URL}/predict_csv",
            files={"file": uploaded_file},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        st.error("⏱️ API timeout. Try a smaller file or increase REQUEST_TIMEOUT.")
        st.stop()
    except Exception as e:
        st.error(f"API call failed: {e}")
        st.stop()

if response.status_code != 200:
    st.error(f"API error ({response.status_code}): {response.text}")
    st.stop()

scored_json = response.json()
df_scored = pd.DataFrame(scored_json)

if df_scored.empty:
    st.warning("No predictions returned.")
    st.stop()

df_scored = ensure_risk_level(df_scored)
df_view = build_business_view(df_scored)

# Metrics
high_count = int((df_view["risk_level"].astype(str).str.contains("High")).sum())
med_count = int((df_view["risk_level"].astype(str).str.contains("Medium")).sum())
low_count = int((df_view["risk_level"].astype(str).str.contains("Low")).sum())

st.subheader("📌 Risk Summary")
c1, c2, c3 = st.columns(3)
c1.metric("🔴 High risk", high_count)
c2.metric("🟠 Medium risk", med_count)
c3.metric("🟢 Low risk", low_count)

# Filter
st.subheader("🎛️ Filters")
high_only = st.checkbox("Show only 🔴 High risk customers", value=False)

if high_only:
    df_selected = df_view[df_view["risk_level"].astype(str).str.contains("High")].copy()
else:
    df_selected = df_view.copy()

st.caption(f"Showing {len(df_selected)} customers (out of {len(df_view)} scored).")

# Table
st.subheader("📋 Clients to Prioritize")
st.dataframe(df_selected, use_container_width=True)

# Export ONLY selection
st.subheader("⬇️ Export")
export_name = "telco_churn_high_risk.csv" if high_only else "telco_churn_prioritization.csv"
csv_bytes = df_selected.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download selected customers (CSV)",
    data=csv_bytes,
    file_name=export_name,
    mime="text/csv",
)
