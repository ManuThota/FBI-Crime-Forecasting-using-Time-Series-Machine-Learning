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

This avoids hardcoding values across files.

=========================================================
"""

# ===============================
# Data Paths
# ===============================
TRAIN_DATA_PATH = "data/raw/Train.csv"
TEST_DATA_PATH = "data/raw/Test.csv"


# ===============================
# Output Paths
# ===============================
PROCESSED_DATA_PATH = "data/processed/"
MODEL_SAVE_PATH = "models/xgboost_model.pkl"


# ===============================
# Split Configuration
# ===============================
TEST_SIZE_RATIO = 0.2


# ===============================
# Random Seed
# ===============================
RANDOM_STATE = 42