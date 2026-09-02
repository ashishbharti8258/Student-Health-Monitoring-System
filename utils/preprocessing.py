import os
import joblib
import pandas as pd

from utils.feature_engineering import add_engineered_features
from utils.config import MODEL_PATH, PREP_PATH


def artifacts_exist() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)


def load_artifacts():
    """Load the trained model + all fitted preprocessing objects."""
    if not artifacts_exist():
        raise FileNotFoundError(
            "Trained model artifacts not found in /models. "
            "Run `python train_model.py` first (after placing your dataset "
            "in data/student_health_data.csv)."
        )
    model = joblib.load(MODEL_PATH)
    prep = joblib.load(PREP_PATH)
    return model, prep


def transform(df_raw: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """
    Apply the full training-time preprocessing pipeline to new data
    (either a full raw dataset or a single-row DataFrame built from a
    Streamlit form). Returns a DataFrame with columns/order matching
    exactly what the model was trained on.
    """
    df = df_raw.copy()

    numerical_features = prep["numerical_features"]
    categorical_features = prep["categorical_features"]
    high_missing_features = prep["high_missing_features"]
    numerical_medians = prep["numerical_medians"]
    dummy_columns = prep["dummy_columns"]
    feature_columns = prep["feature_columns"]
    scaler = prep["scaler"]

    # 1. Missing-value indicator flags (computed BEFORE any imputation)
    for feature in high_missing_features:
        if feature in df.columns:
            df[f"{feature}_isna"] = df[feature].isnull().astype(int)
        else:
            df[f"{feature}_isna"] = 0

    # 2. Categorical NaN -> 'Unknown'
    for feature in categorical_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna("Unknown")
        else:
            df[feature] = "Unknown"

    # 3. Numerical NaN -> TRAINING median (avoids leakage from new data)
    for feature in numerical_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(numerical_medians[feature])
        else:
            df[feature] = numerical_medians[feature]

    # 4. One-hot encode categoricals, aligned to training dummy columns
    columns_before_encoding = set(df.columns)  # includes *_isna flags - excluded from the dummy check below
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=int)
    new_dummy_cols = [c for c in df.columns if c not in columns_before_encoding]
    for col in dummy_columns:
        if col not in df.columns:
            df[col] = 0
    # drop any category dummy that wasn't seen during training (safety net, e.g. an unseen category)
    extra_dummy_cols = [c for c in new_dummy_cols if c not in dummy_columns]
    if extra_dummy_cols:
        df = df.drop(columns=extra_dummy_cols)

    # 5. Scale numeric columns with the training-fitted scaler
    df[numerical_features] = scaler.transform(df[numerical_features])

    # 6. Engineered cross-features
    df = add_engineered_features(df)

    # 7. Final reindex to the exact training feature order
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def predict(df_raw: pd.DataFrame, model, prep: dict):
    """
    Run the full pipeline + model on raw input and return
    (predicted_labels, confidence_per_row, class_probabilities_df).
    """
    X = transform(df_raw, prep)
    pred_encoded = model.predict(X)
    pred_labels = prep["target_encoder"].inverse_transform(pred_encoded)

    proba_df = None
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba_df = pd.DataFrame(proba, columns=prep["target_encoder"].classes_, index=df_raw.index)
        confidence = proba.max(axis=1)

    return pred_labels, confidence, proba_df
