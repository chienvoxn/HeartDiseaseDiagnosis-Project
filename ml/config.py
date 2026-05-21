from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_uci.csv"
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "manual_random_forest.pkl"
REPORT_PATH = PROJECT_ROOT / "ml" / "reports" / "metrics.json"

TARGET_COLUMN = "target"
DROP_COLUMNS = ["id", "dataset", "num", "target"]

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

ZERO_AS_MISSING = ["trestbps", "chol"]
