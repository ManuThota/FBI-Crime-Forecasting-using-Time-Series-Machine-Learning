"""
=========================================================
Project Configuration File
=========================================================

Purpose:
--------
Central place to store:
- File paths
- Constants
- Parameters

=========================================================
"""

# ===============================
# Data Paths
# ===============================
TRAIN_DATA_PATH = "data/raw/Train.csv"
TEST_DATA_PATH = "data/raw/Test.csv"

# ===============================
# Processed Data Paths
# ===============================
PROCESSED_DATA_DIR = "data/processed/"

AGGREGATED_DATA_PATH = PROCESSED_DATA_DIR + "aggregated_encoded.csv"
TIME_SERIES_DATA_PATH = PROCESSED_DATA_DIR + "time_series.csv"

X_TRAIN_PATH = PROCESSED_DATA_DIR + "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR + "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR + "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR + "y_test.csv"

TS_TRAIN_PATH = PROCESSED_DATA_DIR + "train_ts.csv"
TS_TEST_PATH = PROCESSED_DATA_DIR + "test_ts.csv"

# ===============================
# Model Path
# ===============================
MODEL_SAVE_PATH = "models/xgboost_model.pkl"

# ===============================
# Split Configuration
# ===============================
TEST_SIZE_RATIO = 0.2

# ===============================
# Random Seed
# ===============================
RANDOM_STATE = 42