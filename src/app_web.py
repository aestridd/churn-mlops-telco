import os
import requests
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(page_title="Telco Churn – Client Prioritization", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

# =========================
# Session state (for reset)
# =========================
if "search_id" not in st.session_state:
    st.session_state.search_id = ""

if "show_high" not in st.session_state:
    st.session_state.show_high = True
if "show_med" not in st.session_state:
    st.session_state.show_med = True
if "show_low" not in st.session_state:
    st.session_state.show_low = True

if "sort_order" not in st.session_state:
    st.session_state.sort_order = "By customerID (A → Z)"


def reset_filters():
    st.session_state.search_id = ""
    st.session_state.show_high = True
    st.session_state.show_med = True
    st.session_state.show_low = True
    st.session_state.sort_order = "By customerID (A → Z)"


# =========================
# UI
# =========================
st.title("📊 Telco Churn — Client Prioritization Tool")
st.write(
    "Upload a CSV file to score churn risk. Display is **sorted by customerID** by default. "
    "Use the manual filters and search to select the customers you want to see."
)

uploaded_file = st.file_uploader("📂 Upload client CSV", type=["csv"])

if not uploaded_file:
    st.info("⬆️ Upload a CSV file to start scoring clients.")
    st.stop()

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

data = response.json()
summary = data.get("summary", {})
df = pd.DataFrame(data.get("customers", []))

if df.empty:
    st.warning("No predictions returned.")
    st.stop()

# Normalize display
df["churn_probability"] = pd.to_numeric(df["churn_probability"], errors="coerce").round(4)
df["churn_pred"] = pd.to_numeric(df["churn_pred"], errors="coerce").fillna(0).astype(int)
df["customerID"] = df["customerID"].astype(str)

# Convert API risk labels -> emoji labels
risk_map = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}
df["risk_level"] = df["risk_level"].map(risk_map).fillna(df["risk_level"].astype(str))

# =========================
# Summary
# =========================
st.subheader("📌 Risk Summary")
c1, c2, c3 = st.columns(3)
c1.metric("🔴 High risk", int(summary.get("high", 0)))
c2.metric("🟠 Medium risk", int(summary.get("medium", 0)))
c3.metric("🟢 Low risk", int(summary.get("low", 0)))

avg_proba = summary.get("avg_proba", 0.0)
try:
    avg_proba = round(float(avg_proba), 4)
except Exception:
    avg_proba = avg_proba

st.caption(
    f"Scored {int(summary.get('n_scored', len(df)))} customers • "
    f"Avg churn probability: {avg_proba} • "
    f"churn_pred=1 means probability >= business threshold"
)

# =========================
# Filters + Reset
# =========================
st.subheader("🎛️ Filters")
left, right = st.columns([3, 1])
with right:
    st.button("🔄 Reset all filters", on_click=reset_filters)

st.session_state.search_id = st.text_input(
    "🔎 Search customerID",
    value=st.session_state.search_id,
).strip()

col_a, col_b, col_c = st.columns(3)
st.session_state.show_high = col_a.checkbox("🔴 High", value=st.session_state.show_high)
st.session_state.show_med = col_b.checkbox("🟠 Medium", value=st.session_state.show_med)
st.session_state.show_low = col_c.checkbox("🟢 Low", value=st.session_state.show_low)

allowed = set()
if st.session_state.show_high:
    allowed.add("🔴 High")
if st.session_state.show_med:
    allowed.add("🟠 Medium")
if st.session_state.show_low:
    allowed.add("🟢 Low")

sort_options = [
    "By customerID (A → Z)",
    "By churn risk (High → Low)",
    "By churn risk (Low → High)",
]
st.session_state.sort_order = st.selectbox(
    "Sort order",
    options=sort_options,
    index=sort_options.index(st.session_state.sort_order),
)

# Apply filters
df_filtered = df[df["risk_level"].isin(allowed)].copy()

if st.session_state.search_id:
    df_filtered = df_filtered[
        df_filtered["customerID"].str.contains(st.session_state.search_id, case=False, na=False)
    ]

# Apply sorting
if st.session_state.sort_order == "By customerID (A → Z)":
    df_filtered = df_filtered.sort_values("customerID", ascending=True)
elif st.session_state.sort_order == "By churn risk (High → Low)":
    # High first, then probability desc
    order = {"🔴 High": 0, "🟠 Medium": 1, "🟢 Low": 2}
    df_filtered["_order"] = df_filtered["risk_level"].map(order).fillna(99)
    df_filtered = df_filtered.sort_values(["_order", "churn_probability"], ascending=[True, False]).drop(columns="_order")
else:
    # Low first, then probability asc
    order = {"🟢 Low": 0, "🟠 Medium": 1, "🔴 High": 2}
    df_filtered["_order"] = df_filtered["risk_level"].map(order).fillna(99)
    df_filtered = df_filtered.sort_values(["_order", "churn_probability"], ascending=[True, True]).drop(columns="_order")

st.caption(f"Showing {len(df_filtered)} customers (out of {len(df)} scored).")

# =========================
# Table
# =========================
st.subheader("📋 Scored Customers")
st.dataframe(
    df_filtered[["customerID", "churn_probability", "risk_level", "churn_pred"]],
    use_container_width=True,
)

# =========================
# Export
# =========================
st.subheader("⬇️ Export")
csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download current selection (CSV)",
    data=csv_bytes,
    file_name="telco_churn_selection.csv",
    mime="text/csv",
)

import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"
