"""Central configuration: paths, random seed, and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CLUSTER_ASSIGNMENTS_DIR = RESULTS_DIR / "cluster_assignments"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = PROJECT_ROOT / "models"

for _dir in (
    DATA_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    CLUSTER_ASSIGNMENTS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.2

SOCIAL_NETWORK_ADS_PATH = DATA_DIR / "Social_Network_Ads.csv"
MALL_CUSTOMERS_PATH = DATA_DIR / "Mall_Customers.csv"
INSURANCE_PATH = DATA_DIR / "insurance_pre.csv"

CLASSIFICATION_TARGET = "Purchased"
CLASSIFICATION_FEATURES = ["Age", "EstimatedSalary", "Gender"]

CLUSTERING_FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]

REGRESSION_TARGET = "charges"
