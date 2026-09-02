import os
import pandas as pd
import numpy as np
import streamlit as st

from utils.config import DATA_PATH
from utils import visualization as viz

st.set_page_config(page_title="Health Analytics", page_icon="🔬", layout="wide")
st.title("🔬 Health Analytics")


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


df = load_dataset()
if df is None:
    st.warning(f"No dataset found at `{DATA_PATH}`. Add your dataset to unlock this page.")
    st.stop()

# --- Filters (only for columns that actually exist) ---
st.sidebar.header("Filters")
filter_cols = ["gender", "physical_activity_level", "diet_type", "smoking_alcohol", "health_condition"]
filtered = df.copy()
for col in filter_cols:
    if col in df.columns:
        options = sorted(df[col].dropna().unique().tolist())
        selected = st.sidebar.multiselect(col.replace("_", " ").title(), options, default=options)
        if selected:
            filtered = filtered[filtered[col].isin(selected)]

st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** students after filtering.")
st.divider()

st.subheader("Relationship Analysis")
pairs = [
    ("stress_level", "sleep_duration", "box"),
    ("bmi", "step_count", "scatter"),
    ("step_count", "calorie_expenditure", "scatter"),
    ("exercise_duration", "heart_rate", "scatter"),
    ("water_intake", "physical_activity_level", "box_flip"),
    ("sleep_quality", "stress_level", "grouped_bar"),
]

cols = st.columns(2)
i = 0
for a, b, kind in pairs:
    if a not in filtered.columns or b not in filtered.columns:
        continue
    with cols[i % 2]:
        if kind == "scatter":
            color = "health_condition" if "health_condition" in filtered.columns else None
            st.plotly_chart(viz.scatter(filtered, a, b, color=color), use_container_width=True)
        elif kind == "box":
            st.plotly_chart(viz.box_plot(filtered, a, b), use_container_width=True)
        elif kind == "box_flip":
            st.plotly_chart(viz.box_plot(filtered, b, a, title=f"{a.replace('_',' ').title()} by {b.replace('_',' ').title()}"), use_container_width=True)
        elif kind == "grouped_bar":
            st.plotly_chart(viz.grouped_bar(filtered, a, b), use_container_width=True)
    i += 1

st.divider()

st.subheader("Correlation Heatmap")
numeric_cols = filtered.select_dtypes(include=np.number).drop(columns=["id"], errors="ignore").columns.tolist()
if len(numeric_cols) >= 2:
    st.plotly_chart(viz.correlation_heatmap(filtered, numeric_cols), use_container_width=True)
else:
    st.info("Not enough numeric columns available to compute a correlation heatmap.")
