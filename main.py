"""
=========================================================
Main Entry Point of Project
=========================================================

This file runs the pipeline step-by-step.

Currently:
- Loads datasets

Later:
- Will trigger full ML pipeline

=========================================================
"""

# ===============================
# Imports
# ===============================
from src.data_ingestion.load_data import load_datasets
from src.config.config import TRAIN_DATA_PATH, TEST_DATA_PATH


# ===============================
# Main Function
# ===============================
def main():
    """
    Main execution function
    """

    # Step 1: Load Data
    train_df, test_df = load_datasets(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH
    )

    # Display basic info
    print("\nTrain Data Preview:")
    print(train_df.head())

    print("\nTest Data Preview:")
    print(test_df.head())


# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    main()