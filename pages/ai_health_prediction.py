import pandas as pd
import streamlit as st

from utils.model_utils import load_artifacts, predict, get_feature_importance, artifacts_exist
from utils.health_score import compute_health_score
from utils.recommendations import generate_recommendations
from utils.validator import check_dataset_compatibility
from utils import visualization as viz
from utils import db as db_module

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

tab_single, tab_batch = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction"])


def _run_batch(batch_df: pd.DataFrame, source_label: str, download_name: str) -> None:
    """Shared scoring + results rendering for both the DB and CSV batch paths."""
    batch_df.columns = [c.strip().lower().replace(" ", "_") for c in batch_df.columns]
    ok, missing = check_dataset_compatibility(batch_df)
    if not ok:
        st.error("Missing required column(s): " + ", ".join(missing))
        return

    st.success(f"Loaded {len(batch_df):,} rows from {source_label}.")
    if st.button(f"🔮 Run Predictions on {source_label}", use_container_width=True, key=f"run_{download_name}"):
        with st.status("Scoring rows...", expanded=True) as s:
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
            file_name=download_name,
            mime="text/csv",
            use_container_width=True,
        )

        st.subheader("Predicted Category Breakdown")
        st.bar_chart(results["predicted_category"].value_counts())


with tab_batch:
    db_configured = db_module.is_configured()
    source_options = (["🗄️ Live Database (Supabase)", "📤 Upload a CSV"] if db_configured else ["📤 Upload a CSV"])
    batch_source = st.radio("Score which data?", source_options, horizontal=True)

    if batch_source.startswith("🗄️"):
        st.caption("Pulls the most recent rows straight from Supabase/PostgreSQL — no CSV involved.")
        if st.button("🔄 Refresh from Database", key="batch_db_refresh"):
            db_module.fetch_dataframe.clear()
            st.rerun()

        db_df = db_module.fetch_dataframe(limit=5000)
        if db_df is None or db_df.empty:
            st.info("No rows in the database yet — ingest some on the Live Data page first.")
        else:
            _run_batch(db_df, "the live database", "predictions_database.csv")

    else:
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
            else:
                _run_batch(batch_df, f"**{batch_file.name}**", f"predictions_{batch_file.name}")


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