import os
import pandas as pd
import streamlit as st

from utils.config import DATA_PATH
from utils.preprocessing import artifacts_exist

st.set_page_config(
    page_title="Student Health Monitoring System",
    page_icon="🩺",
    layout="wide",
)

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def main():
    st.title("🩺 AI-Powered Student Health Monitoring & Health Risk Prediction")
    st.caption(
        "An interactive dashboard for exploring student health data and predicting "
        "health risk using a trained machine learning model."
    )

    df = load_dataset()
    model_ready = artifacts_exist()

    col1, col2 = st.columns(2)
    with col1:
        if df is not None:
            st.success(f"Dataset loaded: **{df.shape[0]:,}** students, **{df.shape[1]}** columns.")
        else:
            st.warning(
                f"No dataset found at `{DATA_PATH}`. Place your `train.csv` there "
                "(as `student_health_data.csv`) to unlock all pages."
            )
    with col2:
        if model_ready:
            st.success("Trained model artifacts found — AI predictions are ready.")
        else:
            st.warning(
                "No trained model found. Run `python train_model.py` after adding the "
                "dataset to generate `models/health_model.pkl`."
            )

    st.divider()

    st.subheader("What you can do here")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "- **Health Overview** — dataset-wide KPIs, risk distribution, and key histograms\n"
            "- **Student Analysis** — look up an individual student's full health profile "
            "and a 0-100 Health Score"
        )
    with right:
        st.markdown(
            "- **Health Analytics** — explore relationships between health parameters with "
            "filters, scatter plots, box plots, and a correlation heatmap\n"
            "- **AI Health Prediction** — enter health parameters and get a live model "
            "prediction with confidence"
        )

    st.info(
        "**Disclaimer:** This dashboard is an educational / portfolio Data Science project. "
        "The Health Score and AI predictions are analytical tools trained on a Kaggle dataset "
        "— they are **not** medical diagnoses and should never replace professional medical advice.",
        icon="ℹ️",
    )

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("Use the pages above to explore the dashboard.")


if __name__ == "__main__":
    main()
