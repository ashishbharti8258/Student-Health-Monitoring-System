import pandas as pd
import streamlit as st

from utils.preprocessing import load_artifacts, predict, artifacts_exist
from utils.recommendations import generate_recommendations
from utils.health_score import compute_health_score
from utils import visualization as viz

st.set_page_config(page_title="AI Health Prediction", page_icon="🤖", layout="wide")
st.title("🤖 AI Health Risk Prediction")

st.markdown(
    "Enter a student's health parameters below. The system will apply the **same "
    "preprocessing and feature engineering used during training**, run the trained "
    "model, and return a predicted health risk category with confidence."
)

if not artifacts_exist():
    st.error(
        "No trained model found. Run `python train_model.py` (after placing your dataset "
        "in `data/student_health_data.csv`) to enable predictions on this page."
    )
    st.stop()

model, prep = load_artifacts()

numerical_features = prep["numerical_features"]
categorical_features = prep["categorical_features"]
numerical_medians = prep["numerical_medians"]
numerical_ranges = prep["numerical_ranges"]
categorical_categories = prep["categorical_categories"]

NUM_LABELS = {
    "sleep_duration": ("Sleep Duration (hours)", 0.1),
    "heart_rate": ("Resting Heart Rate (bpm)", 1.0),
    "bmi": ("BMI", 0.1),
    "calorie_expenditure": ("Calorie Expenditure (kcal)", 10.0),
    "step_count": ("Step Count", 100.0),
    "exercise_duration": ("Exercise Duration (min)", 1.0),
    "water_intake": ("Water Intake (liters)", 0.1),
}

with st.form("prediction_form"):
    st.subheader("Student Health Parameters")

    inputs = {}
    num_cols = st.columns(2)
    for i, feat in enumerate(numerical_features):
        label, step = NUM_LABELS.get(feat, (feat.replace("_", " ").title(), 1.0))
        lo, hi = numerical_ranges.get(feat, (0.0, 100.0))
        default = numerical_medians.get(feat, (lo + hi) / 2)
        with num_cols[i % 2]:
            inputs[feat] = st.number_input(
                label, min_value=float(lo), max_value=float(hi) * 1.2,
                value=float(default), step=float(step),
            )

    cat_cols = st.columns(2)
    for i, feat in enumerate(categorical_features):
        options = [c for c in categorical_categories.get(feat, []) if c != "Unknown"] or ["Unknown"]
        with cat_cols[i % 2]:
            inputs[feat] = st.selectbox(feat.replace("_", " ").title(), options)

    submitted = st.form_submit_button("🔮 PREDICT HEALTH RISK", use_container_width=True)

if submitted:
    row_df = pd.DataFrame([inputs])

    try:
        labels, confidence, proba_df = predict(row_df, model, prep)
        pred = labels[0]
        conf = confidence[0] if confidence is not None else None

        st.divider()
        st.subheader("AI Health Risk Assessment")

        risk_colors = {"fit": "🟢", "at-risk": "🟡", "unhealthy": "🔴"}
        icon = risk_colors.get(str(pred).lower(), "⚪")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Category", f"{icon} {str(pred).upper()}")
        with c2:
            if conf is not None:
                st.metric("Prediction Confidence", f"{conf*100:.1f}%")

        if proba_df is not None:
            st.markdown("**Class probabilities**")
            st.bar_chart(proba_df.T)

        st.warning(
            "**Disclaimer:** This prediction is generated using a machine learning model "
            "trained on the provided dataset. It is for educational and analytical purposes "
            "only and should not be considered a medical diagnosis.",
            icon="⚠️",
        )

        st.divider()
        score_inputs = {k: v for k, v in inputs.items() if k in [
            "sleep_duration", "step_count", "exercise_duration", "water_intake",
            "heart_rate", "bmi", "stress_level"
        ]}
        score = compute_health_score(score_inputs)
        colA, colB = st.columns([1, 1])
        with colA:
            if score is not None:
                st.plotly_chart(viz.gauge_chart(score), use_container_width=True)
        with colB:
            st.subheader("Personalized Wellness Recommendations")
            for tip in generate_recommendations(inputs):
                st.markdown(f"- {tip}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
