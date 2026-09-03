import time

import pandas as pd
import streamlit as st

from utils import db as db_module
from utils.health_score import compute_health_score
from utils import visualization as viz

st.set_page_config(page_title="Live Data", layout="wide", page_icon="🗄️")
st.title("🗄️ Live Data Ingestion & Metrics")
st.caption("Backed by a cloud PostgreSQL database — records ingested here update dashboard metrics in real time.")

if not db_module.is_configured():
    st.warning(
        "PostgreSQL isn't configured yet. Add your connection details to "
        "`.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example` "
        "as a starting point), then reload this page:"
    )
    st.code(
        '[postgres]\n'
        'host = "your-db-host.example.com"\n'
        'port = 5432\n'
        'dbname = "health_monitoring"\n'
        'user = "your_user"\n'
        'password = "your_password"\n'
        'sslmode = "require"',
        language="toml",
    )
    st.info(
        "Any managed Postgres works: Neon, Supabase, Render, AWS RDS, etc. "
        "The app creates its own `health_records` table automatically on first use."
    )
    st.stop()

# ---- Live metrics ------------------------------------------------------
top_l, top_r = st.columns([4, 1])
with top_l:
    st.subheader("📈 Live Metrics")
with top_r:
    if st.button("🔄 Refresh Now", use_container_width=True):
        db_module.fetch_live_metrics.clear()
        db_module.fetch_dataframe.clear()
        st.rerun()

metrics = db_module.fetch_live_metrics()

if not metrics or not metrics.get("total_records"):
    st.info("No records ingested yet — submit the form below to add the first one.")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{metrics['total_records']:,}")
    m2.metric("Avg Sleep", f"{metrics['avg_sleep']:.1f}h" if metrics["avg_sleep"] is not None else "N/A")
    m3.metric("Avg Heart Rate", f"{metrics['avg_heart_rate']:.0f} bpm" if metrics["avg_heart_rate"] is not None else "N/A")
    m4.metric("Avg BMI", f"{metrics['avg_bmi']:.1f}" if metrics["avg_bmi"] is not None else "N/A")
    if metrics.get("last_ingested_at"):
        st.caption(f"Last record ingested: {metrics['last_ingested_at']}")
    st.caption("Metrics are cached for 15 seconds so the dashboard stays fast under load, then auto-refresh.")

st.divider()

# ---- Ingestion form ------------------------------------------------------
st.subheader("➕ Ingest a New Record")
with st.form("db_ingest_form", border=True):
    c1, c2, c3 = st.columns(3)
    student_id = c1.text_input("Student ID", value="")
    gender = c1.selectbox("Gender", ["Male", "Female", "Other"])
    age = c1.number_input("Age", min_value=10, max_value=100, value=21)

    sleep_duration = c2.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.5)
    heart_rate = c2.number_input("Heart Rate (bpm)", value=75)
    bmi = c2.number_input("BMI", value=22.5)

    calorie_expenditure = c3.number_input("Calorie Expenditure", value=2200)
    step_count = c3.number_input("Step Count", value=8000)
    exercise_duration = c3.number_input("Exercise Duration (min)", value=30)

    c4, c5, c6 = st.columns(3)
    water_intake = c4.number_input("Water Intake (L)", value=2.5)
    stress_level = c5.selectbox("Stress Level", ["Low", "Medium", "High"])
    physical_activity_level = c6.selectbox("Activity Level", ["Sedentary", "Moderate", "Active", "Very Active"])
    health_condition = st.selectbox("Health Condition (if known)", ["Normal", "At Risk", "High Risk", "Unknown"])

    submitted = st.form_submit_button("📥 Ingest Record", use_container_width=True)

if submitted:
    record = {
        "student_id": student_id or None,
        "gender": gender,
        "age": int(age),
        "sleep_duration": sleep_duration,
        "heart_rate": heart_rate,
        "bmi": bmi,
        "calorie_expenditure": calorie_expenditure,
        "step_count": step_count,
        "exercise_duration": exercise_duration,
        "water_intake": water_intake,
        "stress_level": stress_level,
        "physical_activity_level": physical_activity_level,
        "health_condition": None if health_condition == "Unknown" else health_condition,
    }
    try:
        db_module.insert_record(record)
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
    else:
        st.success("Record ingested — metrics above updated.")
        score = compute_health_score(record)
        if score is not None:
            st.plotly_chart(viz.gauge_chart(score, title="This Record's Health Score"), use_container_width=True)
        time.sleep(0.3)
        st.rerun()

st.divider()

# ---- Recent records table ------------------------------------------------
st.subheader("🕓 Recent Records")
recent_df = db_module.fetch_dataframe(limit=200)
if recent_df is None or recent_df.empty:
    st.info("No records to show yet.")
else:
    st.dataframe(recent_df, use_container_width=True, height=350)
    st.download_button(
        "⬇️ Download All Recent Records as CSV",
        data=recent_df.to_csv(index=False).encode("utf-8"),
        file_name="live_health_records.csv",
        mime="text/csv",
    )