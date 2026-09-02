import os
import pandas as pd
import streamlit as st

from utils.config import DATA_PATH
from utils import visualization as viz

st.set_page_config(page_title="Health Overview", page_icon="📊", layout="wide")
st.title("📊 Health Overview Dashboard")

# --- Pull from Session State (Supports Custom Uploads from app.py) ---
if 'current_df' not in st.session_state or st.session_state['current_df'] is None:
    if os.path.exists(DATA_PATH):
        st.session_state['current_df'] = pd.read_csv(DATA_PATH)
    else:
        st.session_state['current_df'] = None

df = st.session_state['current_df']

if df is None:
    st.warning(
        f"No dataset found at `{DATA_PATH}` and no dataset uploaded. "
        "Please upload a dataset on the main page or add your CSV to the `data/` folder."
    )
    st.stop()

# --- Interactive Sidebar Filters for the Overview ---
st.sidebar.markdown("### 🔍 Filter Overview")
TARGET = "health_condition"

if TARGET in df.columns:
    all_risks = df[TARGET].dropna().unique().tolist()
    selected_risks = st.sidebar.multiselect(
        "Filter by Health Risk:",
        options=all_risks,
        default=all_risks
    )
    # Apply filter dynamically
    df_filtered = df[df[TARGET].isin(selected_risks)]
else:
    df_filtered = df.copy()

st.caption(f"Showing metrics for **{len(df_filtered):,}** out of **{len(df):,}** total students.")
st.divider()

# --- KPI cards: only for columns that actually exist ---
kpi_specs = [
    ("heart_rate", "Avg Heart Rate", "bpm"),
    ("sleep_duration", "Avg Sleep Duration", "hrs"),
    ("bmi", "Avg BMI", ""),
    ("step_count", "Avg Step Count", "steps"),
    ("water_intake", "Avg Water Intake", "L"),
    ("exercise_duration", "Avg Exercise Duration", "min"),
    ("calorie_expenditure", "Avg Calorie Expenditure", "kcal"),
]
available_kpis = [(c, label, unit) for c, label, unit in kpi_specs if c in df_filtered.columns]

st.subheader("Key Metrics")
if available_kpis:
    top_row = st.columns(len(available_kpis) + 1)
    top_row[0].metric("Filtered Students", f"{len(df_filtered):,}")
    for i, (col, label, unit) in enumerate(available_kpis, start=1):
        val = df_filtered[col].mean()
        top_row[i].metric(label, f"{val:,.1f} {unit}".strip())
else:
    st.metric("Filtered Students", f"{len(df_filtered):,}")

st.divider()

# --- Risk distribution ---
if TARGET in df_filtered.columns:
    st.subheader("Health Risk Distribution")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(viz.risk_distribution_donut(df_filtered, TARGET), use_container_width=True)
    with c2:
        counts = df_filtered[TARGET].value_counts()
        st.markdown("**Students per risk category**")
        for cat, count in counts.items():
            st.write(f"- **{cat}**: {count:,} ({count/len(df_filtered)*100:.1f}%)")

st.divider()

st.subheader("Distributions & Telemetry Insights")
chart_cols = st.columns(2)
chart_specs = [
    ("sleep_duration", "histogram", "Sleep Duration Distribution"),
    ("stress_level", "bar", "Stress Level Distribution"),
    ("physical_activity_level", "bar", "Physical Activity Distribution"),
    ("bmi", "histogram", "BMI Distribution"),
    ("heart_rate", "histogram", "Heart Rate Distribution"),
]
i = 0
for col, kind, title in chart_specs:
    if col not in df_filtered.columns:
        continue
    target_col = chart_cols[i % 2]
    with target_col:
        if kind == "histogram":
            st.plotly_chart(viz.histogram(df_filtered, col, title=title), use_container_width=True)
        else:
            st.plotly_chart(viz.category_bar(df_filtered, col, title=title), use_container_width=True)
    i += 1