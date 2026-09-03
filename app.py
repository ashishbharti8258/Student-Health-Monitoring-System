import streamlit as st

from utils.data_loader import load_dataset
from utils.model_utils import artifacts_exist

st.set_page_config(page_title="HealthGuard AI", layout="wide", page_icon="🩺")

st.title("🛡️ HealthGuard AI: Student Health Monitoring")
st.markdown("### Predictive Analytics & Risk Stratification Dashboard")

df = load_dataset()
record_count = f"{len(df):,}" if df is not None else "No data loaded"
model_ready = artifacts_exist()

with st.container(border=True):
    col1, col2, col3 = st.columns([2, 1, 1])
    col1.write("This interactive dashboard explores student health data and predicts risk using a trained ML model.")
    col2.metric("Total Records", record_count)
    with col3:
        if model_ready:
            st.success("Model Active")
        else:
            st.warning("Model Not Found")

st.divider()

st.subheader("Dashboard Modules")
c1, c2 = st.columns(2)

# BUG FIX: these switch_page() calls pointed at filenames that don't match
# the actual files in pages/ (e.g. "pages/01_Health_Overview.py" vs the
# real "pages/1_Health_Overview.py"), which raises a
# StreamlitAPIException at click time. Paths below match the real files.
with c1:
    with st.container(border=True):
        st.markdown("#### 📊 Health Overview")
        st.write("Dataset-wide KPIs, risk distribution, and key telemetry histograms.")
        if st.button("Open Overview", key="b1"):
            st.switch_page("pages/1_Health_Overview.py")

    with st.container(border=True):
        st.markdown("#### 🧑‍🎓 Student Analysis")
        st.write("Look up individual student profiles and transparent health scores.")
        if st.button("Open Analysis", key="b2"):
            st.switch_page("pages/2_Student_Analysis.py")

with c2:
    with st.container(border=True):
        st.markdown("#### 🔬 Health Analytics")
        st.write("Explore multi-variable relationships with filters and scatter plots.")
        if st.button("Open Analytics", key="b3"):
            st.switch_page("pages/3_Health_Analytics.py")

    with st.container(border=True):
        st.markdown("#### 🤖 AI Prediction")
        st.write("Enter parameters to get a live model prediction and feature attribution.")
        if st.button("Open AI Predictor", key="b4"):
            st.switch_page("pages/4_AI_Health_Prediction.py")