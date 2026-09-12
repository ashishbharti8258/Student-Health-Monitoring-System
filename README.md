# Student Health Risk Prediction System

## Overview

Welcome to the Student Health Risk Predictor project. 

This application was developed as a Machine Learning mini-project to estimate a student's health-risk category (`fit`, `at-risk`, or `unhealthy`) based on daily habits and physiological data. The app takes inputs like sleep duration, resting heart rate, BMI, step count, and stress levels, processes them through a trained LightGBM classifier, and displays the risk prediction, model confidence, class-probability breakdown, and simple wellness suggestions.

---

## Features

* **ML-Based Classification:** Uses a tuned LightGBM model to classify student health risk.
* **No Train-Serve Mismatch:** Re-applies the exact preprocessing pipeline used during training.
* **Confidence and Probabilities:** Displays model confidence along with a full class-probability bar chart.
* **Custom Health Score:** A 0–100 heuristic lifestyle score for a quick visual overview.
* **Wellness Tips:** Rule-based suggestions tailored to the student's lowest metrics.
* **Lightweight and Fast:** Runs inference on pre-trained model artifacts without dataset uploads or training needed at runtime.

---

## How the Machine Learning Model Works

The dataset was pre-processed and trained offline using LightGBM. The trained model and preprocessing steps are saved in the `models/` folder as `.pkl` files.

During inference, `utils/preprocessing.py` performs the following steps on the input data:

1. **Missing Value Handling:** Fills missing categorical values with `"Unknown"` and numeric values using the training set medians (avoids data leakage).
2. **One-Hot Encoding:** Encodes categorical variables and aligns them with the exact dummy columns present during model training.
3. **Scaling:** Normalizes numeric features using standard scaling (`StandardScaler`).
4. **Feature Engineering:** Computes 6 custom cross-features:
   * `sedentary_bmi_strain`
   * `stress_sleep_deficit`
   * `toxic_cardio_load`
   * `cardiovascular_load`
   * `metabolic_intensity`
   * `hydration_ratio`
5. **Prediction and Confidence:** Passes the cleaned features to LightGBM (`class_weight='balanced'`) to get class probabilities and the final predicted label.

### Model Performance

During model evaluation, LightGBM outperformed other baseline algorithms tested:

* **LightGBM:** ~0.950 Balanced Accuracy
* **XGBoost:** ~0.881 Balanced Accuracy
* **HistGradientBoosting:** ~0.873 Balanced Accuracy
* **RandomForest:** ~0.866 Balanced Accuracy

---

---

## Project Structure

```text
student-health-predictor/
│
├── app.py                  # Streamlit frontend & dashboard UI
├── models/
│   ├── health_model.pkl    # Pre-trained LightGBM model
│   └── preprocessing_objects.pkl # Saved scaler, dummy cols & medians
│
├── utils/
│   ├── config.py           # Paths to model files
│   ├── preprocessing.py    # Data transformation & prediction functions
│   ├── feature_engineering.py # Custom feature creation logic
│   ├── health_score.py     # 0-100 score logic
│   └── recommendations.py  # Wellness tip generator
│
├── requirements.txt        # Required python packages
└── README.md               # Project documentation
