# 🩺 AI-Powered Student Health Risk Prediction System

## Overview

This is a simple, deployable Streamlit application that predicts a student's
health-risk category (`at-risk`, `fit`, or `unhealthy`) from a small set of
lifestyle and physiological inputs. A student's sleep, heart rate, BMI,
activity, diet, stress, and similar details are entered into a form, passed
through the same preprocessing pipeline used during model training, and
scored by a trained LightGBM classifier. The interface then shows the
predicted category, the model's confidence, per-class probabilities, an
optional secondary Health Score, and a handful of rule-based wellness tips.

This project was refactored from a larger analytics/EDA dashboard into a
focused, single-purpose prediction tool suitable for a portfolio or
internship application: one clear ML deployment story, no unrelated
dashboard clutter.

## Features

- ML-based student health-risk prediction (LightGBM)
- Single shared preprocessing pipeline used by both training and the app
  (no train/serve mismatch)
- Confidence score and full class-probability breakdown
- Optional 0–100 Health Score (a simple lifestyle heuristic, not a medical
  score)
- Rule-based wellness suggestions
- No dataset upload and no database required to run — inference only
- Deployable on Streamlit Community Cloud

## Machine Learning

The model and its preprocessing were produced by a separate training
process (not included in this simplified repo) and saved into
`models/preprocessing_objects.pkl` and `models/health_model.pkl`. At
inference time, `utils/preprocessing.py` re-applies the exact same steps:

1. **Missing-value indicators** — `*_isna` flag columns are computed for the
   features that had high missingness during training, before any imputation.
2. **Missing-value handling** — categorical `NaN` → `"Unknown"`; numeric
   `NaN` → the **training median** for that column (never a value computed
   from new data, to avoid leakage).
3. **Categorical encoding** — one-hot encoding (`drop_first=True`), aligned
   to the exact dummy columns seen during training. Any unseen category is
   safely dropped rather than breaking the pipeline.
4. **Numerical scaling** — a `StandardScaler` fitted during training.
5. **Feature engineering** — six derived cross-features computed after
   scaling: `sedentary_bmi_strain`, `stress_sleep_deficit`,
   `toxic_cardio_load`, `cardiovascular_load`, `metabolic_intensity`,
   `hydration_ratio`.
6. **Column reindexing** — the final feature matrix is reindexed to the
   exact column order the model expects.
7. **Model** — a tuned `LightGBMClassifier` (trained with
   `class_weight='balanced'` to handle class imbalance in the target).
8. **Label decoding** — predictions are mapped back to their original class
   names (`at-risk`, `fit`, `unhealthy`) with the saved `LabelEncoder`.

**Reported performance** (from the saved training run, in
`preprocessing_objects.pkl`): the final LightGBM model reached a
**balanced accuracy of ~0.950** on the held-out test split, ahead of the
other models compared during training (HistGradientBoosting ~0.873,
RandomForest ~0.866, XGBoost ~0.881).

## Input Features

| Field | Type | Notes |
|---|---|---|
| Sleep Duration | Numerical | hours/day |
| Heart Rate | Numerical | resting bpm |
| BMI | Numerical | body mass index |
| Calorie Expenditure | Numerical | kcal/day |
| Step Count | Numerical | steps/day |
| Exercise Duration | Numerical | minutes/day |
| Water Intake | Numerical | litres/day |
| Diet Type | Categorical | Balanced / Non-Veg / Veg |
| Stress Level | Categorical | Low / Medium / High |
| Sleep Quality | Categorical | Poor / Average / Good |
| Physical Activity Level | Categorical | Sedentary / Moderate / Active |
| Smoking / Alcohol Use | Categorical | No / Occasional / Yes |
| Gender | Categorical | Female / Male / Other |

## Prediction Classes

The model predicts one of three classes:

- **fit** — indicators broadly consistent with healthy habits
- **at-risk** — some indicators suggest elevated risk
- **unhealthy** — indicators suggest significant health risk

## Project Structure

```
student-health-predictor/
│
├── app.py                          # Streamlit app (UI + orchestration)
│
├── models/
│   ├── health_model.pkl            # trained LightGBM classifier
│   └── preprocessing_objects.pkl   # scaler, encoders, medians, feature order
│
├── utils/
│   ├── __init__.py
│   ├── config.py                   # file paths
│   ├── preprocessing.py            # shared train/inference transform + predict
│   ├── feature_engineering.py      # the 6 engineered cross-features
│   ├── health_score.py             # secondary 0-100 Health Score
│   └── recommendations.py          # rule-based wellness tips
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`. No dataset download or
training step is required — the trained model and preprocessing bundle are
already included in `models/`.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (make sure `models/*.pkl` are committed —
   they are small enough to keep in the repo and are required for inference).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select this repository and branch, and set the main
   file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and
   launch the app automatically.

## Disclaimer

This project is an educational / portfolio machine learning application.
The predicted health-risk category and the Health Score are analytical
outputs of a model trained on a dataset and **are not medical diagnoses**.
They do not constitute medical advice and should never replace consultation
with a qualified healthcare professional.
