import streamlit as st
import numpy as np

from utils.data_loader import get_active_dataset, render_data_uploader, using_uploaded_data, active_source_label
from utils import visualization as viz

st.set_page_config(page_title="Health Overview", layout="wide")

st.title("📊 Health Overview")

render_data_uploader(key_prefix="overview")
df = get_active_dataset()

if using_uploaded_data():
    st.caption(f"📊 Analyzing {active_source_label()} — {len(df):,} rows.")

if df is None:
    st.warning(
        "⚠️ No dataset found at `data/student_health_data.csv`, and nothing uploaded yet. "
        "Showing demo values — upload a CSV in the sidebar, or drop one in the `data/` folder."
    )
    avg_stress, avg_sleep, bmi_median, active_pct = 5.4, 7.2, 22.4, 84
else:
    st.sidebar.header("Global Filters")
    if "health_condition" in df.columns:
        options = sorted(df["health_condition"].dropna().unique().tolist())
        risk_filter = st.sidebar.multiselect("Risk Level", options, default=options)
        if risk_filter:
            df = df[df["health_condition"].isin(risk_filter)]

    avg_stress = df["stress_level"].mean() if "stress_level" in df.columns and np.issubdtype(df["stress_level"].dtype, np.number) else None
    avg_sleep = df["sleep_duration"].mean() if "sleep_duration" in df.columns else None
    bmi_median = df["bmi"].median() if "bmi" in df.columns else None
    active_pct = (
        100 * (df["physical_activity_level"].str.lower() != "sedentary").mean()
        if "physical_activity_level" in df.columns else None
    )

# KPI Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg Stress Level", f"{avg_stress:.1f}/10" if avg_stress is not None else "N/A")
k2.metric("Avg Sleep", f"{avg_sleep:.1f}h" if avg_sleep is not None else "N/A")
k3.metric("BMI Median", f"{bmi_median:.1f}" if bmi_median is not None else "N/A")
k4.metric("Active Students", f"{active_pct:.0f}%" if active_pct is not None else "N/A")

st.divider()

left, right = st.columns(2)

if df is not None:
    with left:
        st.subheader("Risk Distribution")
        if "health_condition" in df.columns:
            st.plotly_chart(viz.risk_distribution_donut(df, "health_condition"), use_container_width=True)
        else:
            st.info("No `health_condition` column found in the dataset.")
    with right:
        st.subheader("Sleep Duration Distribution")
        if "sleep_duration" in df.columns:
            st.plotly_chart(viz.histogram(df, "sleep_duration"), use_container_width=True)
        else:
            st.info("No `sleep_duration` column found in the dataset.")

    st.subheader("Daily Step Count Distribution")
    if "step_count" in df.columns:
        st.plotly_chart(viz.histogram(df, "step_count"), use_container_width=True)
    else:
        st.info("No `step_count` column found in the dataset.")
else:
    with left:
        st.subheader("Risk Distribution")
        st.bar_chart(np.random.randint(10, 100, 3))
    with right:
        st.subheader("Daily Step Count Trend (demo)")
        st.line_chart(np.random.randn(30))