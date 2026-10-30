"""Project-wide constants. Nothing in this module reads or writes files."""
from pathlib import Path

import numpy as np

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
POSITIVE_LABEL = ">50K"
UCI_URL = "https://archive.ics.uci.edu/dataset/2/adult"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

NUM_COLS = ["age", "education-num", "hours-per-week"]
CAP_COLS = ["capital-gain", "capital-loss"]          # heavily skewed
CAT_COLS_ALL = ["workclass", "marital-status", "occupation", "relationship", "native-country"]
CAT_COLS = [c for c in CAT_COLS_ALL if c != "native-country"]
USED_COLS = NUM_COLS + CAP_COLS + CAT_COLS
DROPPED_COLS = ["fnlwgt", "education", "race", "sex"]   # native-country is dropped by choice (see README)
SENSITIVE_COLS = ["sex", "race"]

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
THRESHOLD_GRID = np.round(np.arange(0.20, 0.71, 0.01), 2)
