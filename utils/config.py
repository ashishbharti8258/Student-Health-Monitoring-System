from pathlib import Path
import streamlit as st

# utils/config.py lives inside the "utils" folder, so the project root is
# ONE level above that (parent.parent), not parent. Using just `.parent`
# was pointing DATA_PATH/MODEL_PATH at "utils/data" and "utils/models",
# which don't exist - that's why the app couldn't find the dataset/model.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_PATH = DATA_DIR / "student_health_data.csv"
MODEL_PATH = MODELS_DIR / "health_model.pkl"
PREP_PATH = MODELS_DIR / "preprocessing_objects.pkl"

st.set_page_config(
    page_title="AI-Powered Student Health Monitoring",
    page_icon="🩺",
    layout="wide"
)