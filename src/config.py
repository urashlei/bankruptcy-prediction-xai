from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "3year.csv"

MODEL_DIR = ROOT_DIR / "models" / "reconstructed"

SHAP_DIR = ROOT_DIR / "results" / "shap"

STABILITY_DIR = ROOT_DIR / "results" / "stability"


# =====================================================
# DATA CONFIGURATION
# =====================================================

TARGET = "class"

NON_BANKRUPT = 0
BANKRUPT = 1

N_FEATURES = 64

N_SPLITS = 5

RANDOM_STATE = 42


# =====================================================
# RANDOM FOREST CONFIGURATION
# Extracted from RapidMiner RF processes
# =====================================================

RF_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 10,
    "min_samples_leaf": 2,
    "min_samples_split": 4,
    "max_features": "sqrt",
    "n_jobs": -1
}


# =====================================================
# H2O GRADIENT BOOSTED TREES CONFIGURATION
# Based on supplied RapidMiner settings
# =====================================================

GBT_PARAMS = {
    "ntrees": 100,
    "max_depth": 5,
    "learn_rate": 0.1,
    "min_rows": 10,
    "min_split_improvement": 0.00001,
    "nbins": 20,
    "sample_rate": 1.0,
    "stopping_rounds": 0,
    "seed": RANDOM_STATE
}


# =====================================================
# SMOTE CONFIGURATION
# =====================================================

SMOTE_PARAMS = {
    "k_neighbors": 5,
    "sampling_strategy": 1.0,
    "random_state": RANDOM_STATE
}


# =====================================================
# METACOST CONFIGURATION
# =====================================================

COST_MATRIX = [
    [0.0, 20.0],
    [1.0, 0.0]
]

METACOST_ITERATIONS = 10

METACOST_SEED = 42


# =====================================================
# SHAP CONFIGURATION
# =====================================================

# Maximum validation samples used to calculate SHAP
# None means using the entire validation fold.

MAX_EXPLAIN_ROWS = 1500


# =====================================================
# MODEL NAMES
# =====================================================

MODEL_NAMES = [
    "RF_Baseline",
    "RF_SMOTE",
    "RF_CostSensitive",

    "GBT_Baseline",
    "GBT_SMOTE",
    "GBT_CostSensitive"
]