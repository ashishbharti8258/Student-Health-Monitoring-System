
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "health_model.pkl"
PREP_PATH = MODELS_DIR / "preprocessing_objects.pkl"
