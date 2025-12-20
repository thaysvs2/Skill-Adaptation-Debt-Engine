from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = BASE_DIR / "data" / "raw" / "HR-Employee-Attrition.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESSED = DATA_PROCESSED_DIR / "attrition_processed.parquet"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "attrition_logreg_pipeline.joblib"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_NARRATIVES_DIR = REPORTS_DIR / "narratives"
REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"

ID_COL = "EmployeeNumber"
TARGET_COL = "Attrition"

DROP_COLS = [
    "EmployeeCount",  # constant in many IBM-style sets
    "StandardHours",  # constant
]

# Peer grouping for "context"
PEER_GROUP_COLS = ["Department", "JobRole", "JobLevel"]

# Controllable levers for counterfactual suggestions
CONTROLLABLE = {
    "OverTime": ["No", "Yes"],               # encourage No
    "WorkLifeBalance": [1, 2, 3, 4],         # higher is better
    "JobSatisfaction": [1, 2, 3, 4],
    "EnvironmentSatisfaction": [1, 2, 3, 4],
    "RelationshipSatisfaction": [1, 2, 3, 4],
    "TrainingTimesLastYear": [0, 1, 2, 3, 4, 5, 6],
}
