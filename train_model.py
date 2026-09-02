import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix,
)

# Make `utils` importable when running as a plain script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.feature_engineering import add_engineered_features
from utils.config import DATA_PATH, MODELS_DIR

TARGET = "health_condition"
ID_COL = "id"

# Exact column groups from the notebook
HIGH_MISSING_FEATURES = ["stress_level", "sleep_duration", "sleep_quality", "calorie_expenditure", "water_intake"]
CATEGORICAL_FEATURES = ["diet_type", "stress_level", "sleep_quality", "physical_activity_level", "smoking_alcohol", "gender"]
NUMERICAL_FEATURES = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration", "water_intake"]


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find dataset at {DATA_PATH}.\n"
            "Download the Kaggle 'Predicting Student Health Risk' train.csv "
            "and save it as data/student_health_data.csv, then re-run this script."
        )
    data = pd.read_csv(DATA_PATH)

    required = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing)}. "
            "This pipeline is built specifically for the notebook's schema — "
            "please confirm you're using the correct train.csv."
        )
    return data


def build_training_frame(data: pd.DataFrame):
    """Mirrors notebook cells 8-11 exactly (missing handling -> encoding ->
    scaling -> engineered features), fitting all transformers on this data."""
    df = data.copy()

    # --- Missing value flags + imputation (cell 8) ---
    for feature in HIGH_MISSING_FEATURES:
        df[f"{feature}_isna"] = df[feature].isnull().astype(int)

    for feature in CATEGORICAL_FEATURES:
        df[feature] = df[feature].fillna("Unknown")

    numerical_medians = {feat: float(data[feat].median()) for feat in NUMERICAL_FEATURES}
    for feature in NUMERICAL_FEATURES:
        df[feature] = df[feature].fillna(numerical_medians[feature])

    # --- Target + categorical encoding (cell 9) ---
    target_encoder = LabelEncoder()
    df[TARGET] = target_encoder.fit_transform(df[TARGET].astype(str))

    categorical_categories = {feat: sorted(df[feat].unique().tolist()) for feat in CATEGORICAL_FEATURES}

    columns_before_encoding = set(df.columns)  # includes the *_isna flags, so they aren't mistaken for dummies
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)
    dummy_columns = [c for c in df.columns if c not in columns_before_encoding]

    # --- Scaling (cell 10) ---
    scaler = StandardScaler()
    df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])

    # --- Engineered features (cell 11) ---
    df = add_engineered_features(df)

    numerical_ranges = {feat: (float(data[feat].min()), float(data[feat].max())) for feat in NUMERICAL_FEATURES}

    prep = {
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "high_missing_features": HIGH_MISSING_FEATURES,
        "numerical_medians": numerical_medians,
        "numerical_ranges": numerical_ranges,
        "categorical_categories": categorical_categories,
        "dummy_columns": dummy_columns,
        "scaler": scaler,
        "target_encoder": target_encoder,
    }
    return df, prep


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    data = load_data()
    print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns.")

    print("Running preprocessing + feature engineering pipeline...")
    df, prep = build_training_frame(data)

    X = df.drop(columns=[ID_COL, TARGET], errors="ignore")
    y = df[TARGET]
    prep["feature_columns"] = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape}")

    # --- Compare candidate models (cell 14) ---
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    candidates = {
        "HistGradientBoost": HistGradientBoostingClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
    }
    try:
        from lightgbm import LGBMClassifier
        candidates["LightGBM"] = LGBMClassifier(class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
    except ImportError:
        print("lightgbm not installed - skipping from comparison (final model still tries to use it below).")
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = XGBClassifier(scale_pos_weight=1, n_jobs=-1, random_state=42)
    except ImportError:
        print("xgboost not installed - skipping from comparison.")

    print("\n--- Model comparison (balanced accuracy) ---")
    results = {}
    for name, clf in candidates.items():
        start = time.time()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        score = balanced_accuracy_score(y_test, preds)
        results[name] = score
        print(f"{name}: {score:.4f} ({time.time() - start:.1f}s)")

    # --- Final tuned model: LightGBM (best performer / deployed model in notebook) ---
    print("\nTraining final deployment model (LightGBM, tuned)...")
    try:
        from lightgbm import LGBMClassifier
        final_model = LGBMClassifier(
            objective="multiclass",
            num_class=len(prep["target_encoder"].classes_),
            class_weight="balanced",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("lightgbm not available in this environment - falling back to RandomForest "
              "so the pipeline still produces a working model. Install lightgbm and re-run "
              "for the notebook's actual best model.")
        from sklearn.ensemble import RandomForestClassifier
        final_model = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42)

    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    print("\n--- Final Model Evaluation ---")
    class_names = prep["target_encoder"].classes_
    print(classification_report(y_test, y_pred, target_names=class_names))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    prep["test_balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
    prep["model_comparison"] = results
    prep["confusion_matrix"] = cm.tolist()
    prep["class_names"] = class_names.tolist()

    joblib.dump(final_model, os.path.join(MODELS_DIR, "health_model.pkl"))
    joblib.dump(prep, os.path.join(MODELS_DIR, "preprocessing_objects.pkl"))
    print(f"\nSaved model + preprocessing objects to {MODELS_DIR}/")

    
    data.to_csv(DATA_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
