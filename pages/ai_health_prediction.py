import pandas as pd
import streamlit as st

from utils.model_utils import load_artifacts, predict, get_feature_importance, artifacts_exist
from utils.health_score import compute_health_score
from utils.recommendations import generate_recommendations
from utils.validator import check_dataset_compatibility
from utils import visualization as viz

st.set_page_config(page_title="AI Prediction", layout="wide")
st.title("🤖 AI Health Risk Prediction")

if not artifacts_exist():
    st.error(
        "⚠️ Trained model artifacts not found in `models/`. "
        "Run `python train_model.py` first (after placing your dataset in "
        "`data/student_health_data.csv`) to generate `health_model.pkl` and "
        "`preprocessing_objects.pkl`."
    )
    st.stop()

tab_single, tab_batch = st.tabs(["🔮 Single Prediction", "📤 Analyze Your Own Data (Batch)"])

with tab_batch:
    st.markdown("### Upload a CSV to score every row with the model")
    st.caption(
        "Needs these columns (case/space-insensitive): sleep_duration, heart_rate, bmi, "
        "calorie_expenditure, step_count, exercise_duration, water_intake."
    )
    batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_predict_uploader")

    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            batch_df = None

        if batch_df is not None:
            batch_df.columns = [c.strip().lower().replace(" ", "_") for c in batch_df.columns]
            ok, missing = check_dataset_compatibility(batch_df)
            if not ok:
                st.error("Missing required column(s): " + ", ".join(missing))
            else:
                st.success(f"Loaded {len(batch_df):,} rows from **{batch_file.name}**.")
                if st.button("🔮 Run Predictions on This File", use_container_width=True):
                    with st.status("Scoring uploaded rows...", expanded=True) as s:
                        model, prep = load_artifacts()
                        st.write(f"Running inference on {len(batch_df):,} rows...")
                        pred_labels, confidence, proba_df = predict(batch_df, model, prep)
                        s.update(label="Batch inference complete", state="complete")

                    results = batch_df.copy()
                    results["predicted_category"] = pred_labels
                    if confidence is not None:
                        results["confidence"] = confidence

                    st.subheader("Results")
                    st.dataframe(results, use_container_width=True)

                    st.download_button(
                        "⬇️ Download Results as CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name=f"predictions_{batch_file.name}",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    if "predicted_category" in results.columns:
                        st.subheader("Predicted Category Breakdown")
                        st.bar_chart(results["predicted_category"].value_counts())


with tab_single:
    with st.form("prediction_input", border=True):
        st.markdown("### Input Student Biometrics")
        c1, c2 = st.columns(2)

        sleep = c1.slider("Sleep Duration (Hours)", 0.0, 12.0, 7.5)
        stress = c1.select_slider("Stress Level", options=["Low", "Medium", "High"], value="Medium")
        heart_rate = c1.number_input("Heart Rate (bpm)", value=75)
        bmi = c2.number_input("BMI", value=22.5)
        activity = c2.selectbox("Activity Level", ["Sedentary", "Moderate", "Active", "Very Active"])
        step_count = c2.number_input("Daily Step Count", value=8000)
        exercise_duration = c2.number_input("Exercise Duration (min/day)", value=30)
        water_intake = c1.number_input("Water Intake (L/day)", value=2.5)
        calorie_expenditure = c2.number_input("Calorie Expenditure", value=2200)

        submit = st.form_submit_button("🔮 RUN AI INFERENCE", use_container_width=True)

    if submit:
        inputs = {
            "sleep_duration": sleep,
            "stress_level": stress,
            "heart_rate": heart_rate,
            "bmi": bmi,
            "physical_activity_level": activity,
            "step_count": step_count,
            "exercise_duration": exercise_duration,
            "water_intake": water_intake,
            "calorie_expenditure": calorie_expenditure,
        }

        with st.status("Running the trained model...", expanded=True) as s:
            st.write("Loading model + preprocessing artifacts...")
            model, prep = load_artifacts()

            st.write("Preprocessing input features...")
            raw_df = pd.DataFrame([inputs])

            st.write("Running inference...")
            pred_labels, confidence, proba_df = predict(raw_df, model, prep)

            s.update(label="Inference Complete", state="complete")

        st.divider()

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted Health Category", str(pred_labels[0]))
            st.success(f"The model predicts **{pred_labels[0]}** for this student profile.")
        with res_col2:
            conf = confidence[0] if confidence is not None else None
            st.metric("Prediction Confidence", f"{conf*100:.1f}%" if conf is not None else "N/A")
            if conf is not None:
                st.progress(min(max(conf, 0.0), 1.0))

        if proba_df is not None:
            st.subheader("Class Probabilities")
            st.bar_chart(proba_df.iloc[0])

        st.subheader("Model Feature Importance")
        st.caption("Top features driving predictions for this model overall (global importance):")
        imp_df = get_feature_importance(model, prep, top_n=8)
        if not imp_df.empty:
            st.bar_chart(imp_df.set_index("feature")["importance"])
        else:
            st.info("This model type doesn't expose feature importances.")

        st.divider()
        score = compute_health_score(inputs)
        if score is not None:
            st.subheader("Composite Health Score")
            st.plotly_chart(viz.gauge_chart(score), use_container_width=True)

        st.subheader("💡 Personalized Recommendations")
        for tip in generate_recommendations(inputs):
            st.write(f"- {tip}")