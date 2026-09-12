import streamlit as st
import pandas as pd
from utils.preprocessing import artifacts_exist, load_artifacts, predict
from utils.health_score import compute_health_score
from utils.recommendations import generate_recommendations

# Full-width page layout
st.set_page_config(
    page_title="Student Health Risk Predictor", 
    page_icon="🩺", 
    layout="wide"
)

# ==========================================
# LEFT SIDEBAR: DASHBOARD INFO & CREATOR DETAILS
# ==========================================
with st.sidebar:
    st.header("ℹ️ About the Project")
    st.write(
        "This Student Health Risk Predictor estimates health-risk categories "
        "using machine learning based on lifestyle and physiological data."
    )
    
    st.divider()
    
    st.subheader("📌 How to Use")
    st.markdown(
        "1. Enter physiological & lifestyle parameters in the **Input Form**.\n"
        "2. Click on **🔮 Predict Health Risk**.\n"
        "3. Review predictions, confidence scores, and personal suggestions."
    )
    
    st.divider()
    
    st.subheader("⚙️ Key Features Analyzed")
    st.markdown(
        "- **Physical Metrics:** Sleep duration, Resting HR, BMI, Step Count, Exercise\n"
        "- **Habits:** Diet Type, Water Intake, Smoking/Alcohol\n"
        "- **Wellbeing:** Stress Level & Sleep Quality"
    )
    
    st.divider()
    
    st.subheader("🎯 Risk Categories")
    st.markdown(
        "- **Low Risk:** Good habit metrics\n"
        "- **Moderate Risk:** Minor deficit in activity/sleep\n"
        "- **High Risk:** Elevated HR, stress, or low activity"
    )

    st.divider()

    
# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================
st.title("🩺 Student Health Risk Predictor Dashboard")
st.divider()

# Check for model files
if not artifacts_exist():
    st.error("⚠️ Trained model files not found in models/ folder! Please check.")
    st.stop()

# --- 2-COLUMN MAIN LAYOUT ---
col_input, col_result = st.columns([1, 1], gap="large")

# ==========================================
# COLUMN 1: INPUT FORM
# ==========================================
with col_input:
    st.subheader("📋 Enter Student Information")
    
    with st.form("prediction_form"):
        st.markdown("**Numerical Parameters**")
        c1, c2 = st.columns(2)
        with c1:
            sleep_duration = st.slider("Sleep Duration (hrs/day)", 0.0, 12.0, 7.0, 0.1)
            heart_rate = st.number_input("Resting HR (bpm)", min_value=40, max_value=180, value=75)
            bmi = st.number_input("BMI", min_value=10.0, max_value=45.0, value=22.5, step=0.1)
            calorie_expenditure = st.number_input("Calories (kcal/day)", min_value=800, max_value=5000, value=2200)
        with c2:
            step_count = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000, step=100)
            exercise_duration = st.number_input("Exercise (min/day)", min_value=0, max_value=180, value=30)
            water_intake = st.number_input("Water Intake (L/day)", min_value=0.0, max_value=6.0, value=2.5, step=0.1)

        st.markdown("**Lifestyle Parameters**")
        c3, c4 = st.columns(2)
        with c3:
            diet_type = st.selectbox("Diet Type", ["Balanced", "Non-Veg", "Veg"])
            stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)
            sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"], index=1)
        with c4:
            physical_activity_level = st.selectbox("Activity Level", ["Sedentary", "Moderate", "Active"], index=1)
            smoking_alcohol = st.selectbox("Smoking / Alcohol", ["No", "Occasional", "Yes"])
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])

        submitted = st.form_submit_button("🔮 Predict Health Risk", use_container_width=True)

# ==========================================
# COLUMN 2: PREDICTION RESULTS
# ==========================================
with col_result:
    st.subheader("🤖 Prediction Results & Analytics")

    if submitted:
        # Dictionary mappings
        diet_map = {"Balanced": "balanced", "Non-Veg": "non-veg", "Veg": "veg"}
        stress_map = {"Low": "low", "Medium": "medium", "High": "high"}
        sleep_map = {"Poor": "poor", "Average": "average", "Good": "good"}
        activity_map = {"Sedentary": "sedentary", "Moderate": "moderate", "Active": "active"}
        smoke_map = {"No": "no", "Occasional": "occasional", "Yes": "yes"}
        gender_map = {"Female": "female", "Male": "male", "Other": "other"}

        inputs = {
            "sleep_duration": sleep_duration,
            "heart_rate": heart_rate,
            "bmi": bmi,
            "calorie_expenditure": calorie_expenditure,
            "step_count": step_count,
            "exercise_duration": exercise_duration,
            "water_intake": water_intake,
            "diet_type": diet_map[diet_type],
            "stress_level": stress_map[stress_level],
            "sleep_quality": sleep_map[sleep_quality],
            "physical_activity_level": activity_map[physical_activity_level],
            "smoking_alcohol": smoke_map[smoking_alcohol],
            "gender": gender_map[gender],
        }

        try:
            # Load model and run inference
            model, prep = load_artifacts()
            raw_df = pd.DataFrame([inputs])
            pred_label, confidence, proba_series = predict(raw_df, model, prep)
            score = compute_health_score(inputs)

            # Top Row KPI Cards
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted Condition", pred_label.upper())
            with m2:
                st.metric("Model Confidence", f"{confidence * 100:.1f}%" if confidence is not None else "N/A")

            if score is not None:
                st.metric("Overall Health Score", f"{score} / 100")

            st.divider()

            # Probabilities section
            if proba_series is not None:
                st.markdown("**📊 Prediction Probabilities**")
                for class_name, prob in proba_series.items():
                    st.write(f"**{class_name.title()}** — {prob * 100:.1f}%")
                    st.progress(float(prob))

            st.divider()

            # Wellness Suggestions
            st.markdown("**💡 Recommendations**")
            for tip in generate_recommendations(inputs):
                st.write(f"- {tip}")

        except Exception as e:
            st.error(f"⚠️ Error running prediction: {e}")

    else:
        st.info("👈 Fill out the details in the left panel and click **Predict Health Risk** to display the dashboard analytics.")

st.divider()
st.caption("**Project Disclaimer:** Developed for academic submission and portfolio demonstration purposes.")