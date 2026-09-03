
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
 
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
 
DATA_PATH = DATA_DIR / "student_health_data.csv"
MODEL_PATH = MODELS_DIR / "health_model.pkl"
PREP_PATH = MODELS_DIR / "preprocessing_objects.pkl"
 