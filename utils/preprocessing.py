import os
import joblib
import pandas as pd
import streamlit as st

from utils.config import MODEL_PATH, PREP_PATH
from utils.feature_engineering import add_engineered_features


# check if models exist
def artifacts_exist():
    return os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)


# load saved model and preprocessing dict
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not artifacts_exist():
        raise FileNotFoundError("Model files not found! Please check models/ folder.")
    
    model = joblib.load(MODEL_PATH)
    prep = joblib.load(PREP_PATH)
    return model, prep


# preprocessing function to transform input dataframe
def transform(df_raw, prep):
    df = df_raw.copy()

    # extract stuff from saved prep dict
    num_cols = prep["numerical_features"]
    cat_cols = prep["categorical_features"]
    missing_cols = prep["high_missing_features"]
    medians = prep["numerical_medians"]
    dummy_cols = prep["dummy_columns"]
    final_feature_order = prep["feature_columns"]
    scaler = prep["scaler"]

    # 1. missing flags
    for col in missing_cols:
        if col in df.columns:
            df[f"{col}_isna"] = df[col].isnull().astype(int)
        else:
            df[f"{col}_isna"] = 0

    # 2. fill missing categoricals with Unknown
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = "Unknown"

    # 3. fill numerical nulls with median
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(medians[col])
        else:
            df[col] = medians[col]

    # 4. one-hot encoding
    cols_before = set(df.columns)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    new_dummies = [c for c in df.columns if c not in cols_before]
    for col in dummy_cols:
        if col not in df.columns:
            df[col] = 0
            
    # drop extra columns if any
    extra_cols = [c for c in new_dummies if c not in dummy_cols]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    # 5. scale numerical columns
    df[num_cols] = scaler.transform(df[num_cols])

    # 6. feature engineering
    df = add_engineered_features(df)

    # 7. match exact training columns
    df = df.reindex(columns=final_feature_order, fill_value=0)

    return df


# main predict function
def predict(df_raw, model, prep):
    X = transform(df_raw, prep)
    
    # get label prediction
    pred_encoded = model.predict(X)
    pred_label = prep["target_encoder"].inverse_transform(pred_encoded)[0]

    proba_series = None
    confidence = None
    
    # check if model supports probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba_series = pd.Series(proba[0], index=prep["target_encoder"].classes_)
        confidence = float(proba[0].max())

    return pred_label, confidence, proba_series