# 🩺 AI-Powered Student Health Monitoring & Health Risk Prediction System

An interactive Streamlit dashboard that analyzes student health parameters, visualizes
health trends, and uses a trained machine learning model to predict student health risk.
Built on top of an original Jupyter notebook (EDA → preprocessing → feature engineering →
model comparison → LightGBM classifier), this project turns that notebook into a
deployable, professional dashboard.

## Overview

The dashboard has four pages: a dataset-wide overview, a per-student profile with a
transparent 0–100 Health Score, exploratory analytics with filters, and a live AI risk
prediction tool. The prediction page reuses the **exact same preprocessing and feature
engineering pipeline** used during training, so there is no train/serve skew.

## Problem Statement

Given a set of student lifestyle and physiological measurements (sleep, heart rate, BMI,
activity, stress, diet, etc.), predict whether a student is `at-risk`, `fit`, or
`unhealthy`, and surface that information through an accessible, explorable dashboard
rather than a raw notebook.

## Features

- 📊 Dataset-wide KPI cards and interactive Plotly distribution charts
- 🧑‍🎓 Individual student lookup with full health profile and a documented Health Score
- 🔬 Filterable relationship analytics (scatter, box plots, correlation heatmap)
- 🤖 Live AI health risk prediction with class probabilities / confidence
- 💡 Rule-based, transparent wellness recommendations
- 🔁 Single shared preprocessing pipeline used by both training and inference

## Dataset

Source: Kaggle *Predicting Student Health Risk* (Playground Series S6E7), inspired by the
College Student Health Behavior Dataset. Target column: `health_condition`
(`at-risk` / `fit` / `unhealthy`).

| Type | Columns |
|---|---|
| Numerical | `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake` |
| Categorical | `diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender` |
| Other | `id`, `health_condition` (target) |

> The dataset itself is **not bundled** with this project (Kaggle competition data
> shouldn't be redistributed). Download `train.csv` and save it as
> `data/student_health_data.csv` before running training.

## Machine Learning Pipeline

Reproduced from the original notebook, in `train_model.py` and `utils/`:

1. **Missing values** — indicator flags (`*_isna`) for the 5 high-missing columns, then
   categorical NaN → `"Unknown"`, numeric NaN → **training median** (no leakage).
2. **Encoding** — `LabelEncoder` on the target; one-hot encoding (`drop_first=True`) on
   the 6 categorical columns.
3. **Scaling** — `StandardScaler` on the 7 numeric columns.
4. **Feature engineering** — six cross-features computed *after* scaling:
   `sedentary_bmi_strain`, `stress_sleep_deficit`, `toxic_cardio_load`,
   `cardiovascular_load`, `metabolic_intensity`, `hydration_ratio`.
5. **Split** — stratified 80/20 train/test split, `random_state=42`.
6. **Model comparison** — LightGBM, XGBoost, HistGradientBoosting, RandomForest, scored
   by balanced accuracy (the target is heavily imbalanced).
7. **Final model** — tuned `LightGBMClassifier` (`class_weight='balanced'`,
   `n_estimators=600`, `learning_rate=0.05`, `num_leaves=31`), matching the model actually
   deployed in the original notebook's submission cells.

All fitted objects (scaler, label encoder, learned categories, medians, feature order) are
saved to `models/preprocessing_objects.pkl` and reused verbatim at prediction time via
`utils/preprocessing.py::transform()` — the single function both training and the
Streamlit app call, so there's exactly one preprocessing implementation.

## Health Score Methodology

The 0–100 **Student Health Score** (see `utils/health_score.py`) is a simple, documented,
weighted composite of the student's *raw* inputs (sleep, steps, exercise, hydration,
resting heart rate, BMI, stress) against generic healthy-range reference points. It is
**independent of the ML model** and is explicitly a dashboard indicator, not a medical
score — this is stated on every page that shows it.

## Dashboard Screens

1. **Health Overview** — KPI cards, risk distribution donut, and histograms/bar charts
   for sleep, stress, activity, BMI, and heart rate.
2. **Individual Student Analysis** — full profile, Health Score gauge, and AI-predicted
   risk for a selected student.
3. **Health Analytics** — filterable scatter/box plots and a correlation heatmap across
   numeric features.
4. **AI Health Prediction** — a form for entering a hypothetical student's parameters,
   producing a predicted category, confidence, and rule-based recommendations.

## Project Structure

```
student-health-monitoring/
│
├── app.py                          # Main entry point / landing page
├── train_model.py                  # Reproducible training + artifact-saving script
│
├── pages/
│   ├── 1_Health_Overview.py
│   ├── 2_Student_Analysis.py
│   ├── 3_Health_Analytics.py
│   └── 4_AI_Health_Prediction.py
│
├── models/
│   ├── health_model.pkl            # generated by train_model.py
│   └── preprocessing_objects.pkl   # generated by train_model.py
│
├── data/
│   └── student_health_data.csv     # place your Kaggle train.csv here
│
├── utils/
│   ├── config.py                   # shared paths
│   ├── preprocessing.py            # shared train/inference pipeline
│   ├── feature_engineering.py      # the 6 engineered features
│   ├── health_score.py             # 0-100 Health Score
│   ├── recommendations.py          # rule-based wellness tips
│   └── visualization.py            # Plotly chart builders
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation & Setup (Windows / VS Code)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your dataset
#    Download train.csv from the Kaggle competition and save it as:
#    data/student_health_data.csv

# 4. Train the model (creates models/health_model.pkl + preprocessing_objects.pkl)
python train_model.py

# 5. Run the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Technologies Used

- **Python 3.10+**
- **Streamlit** — dashboard framework
- **Plotly** — interactive charts
- **scikit-learn** — preprocessing, splitting, evaluation
- **LightGBM / XGBoost** — gradient boosting classifiers
- **pandas / numpy** — data handling
- **joblib** — model persistence

## Model Performance

Run `python train_model.py` to reproduce the model comparison and evaluation metrics
(balanced accuracy, classification report, confusion matrix) on your own copy of the
dataset — actual numbers depend on the data you train against, so they aren't hardcoded
here. The script prints a full comparison table and saves it into
`preprocessing_objects.pkl["model_comparison"]` for reference.

## Future Improvements

- Hyperparameter tuning via Optuna (scaffolded but commented out in the original notebook)
- SMOTETomek or similar resampling to address class imbalance beyond `class_weight='balanced'`
- Model explainability (SHAP) on the prediction page
- User authentication for multi-student longitudinal tracking

## Disclaimer

This project is an educational / portfolio Data Science and Machine Learning
application. The Health Score and AI-predicted health risk are analytical outputs of a
model trained on a Kaggle dataset. **They are not medical diagnoses, do not constitute
medical advice, and should never replace consultation with a qualified healthcare
professional.**
