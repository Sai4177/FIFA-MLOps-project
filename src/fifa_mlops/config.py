"""Central configuration: paths, constants and names used by every experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---- dataset versions: Raw -> Processed -> Feature-engineered -------------------
RAW_PATH = ROOT / "data" / "raw" / "fifa_ranking_2022-10-06.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "fifa_processed_v2.csv"
FEATURES_PATH = ROOT / "data" / "features" / "fifa_features_v3.csv"
STATS_PATH = ROOT / "data" / "features" / "feature_stats_v3.json"
VERSIONS_PATH = ROOT / "data" / "versions.json"
REPORTS_DIR = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"

# ---- ML problem -----------------------------------------------------------------
# Predict a team's CURRENT FIFA tier from the PREVIOUS ranking snapshot only
# (previous_rank, previous_points, association).  Current rank/points are used only
# to build the label, never as features (no target leakage).
TARGET = "tier"
TIER_ORDER = ["Elite", "Strong", "Mid", "Developing"]
RAW_FEATURES = ["previous_rank", "previous_points", "association"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---- MLflow ---------------------------------------------------------------------
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
MLFLOW_ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()
EXPERIMENT_NAME = "FIFA_Tier_Classification"
REGISTERED_MODEL_NAME = "FIFA_Tier_Classifier"
CHAMPION_ALIAS = "champion"
