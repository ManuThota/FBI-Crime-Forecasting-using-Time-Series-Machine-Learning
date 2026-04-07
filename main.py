"""
=========================================================
Main Entry Point of Project
=========================================================

Purpose:
--------
This file controls the execution of the entire pipeline.

=========================================================
"""

# ===============================
# Imports
# ===============================
from src.data_ingestion.load_data import load_datasets
from src.data_cleaning.cleaning import clean_data
from src.config.config import TRAIN_DATA_PATH, TEST_DATA_PATH


# ===============================
# Main Function
# ===============================
def main():
    """
    Main execution pipeline
    """

    print("\n Starting FBI Crime Forecasting Pipeline...\n")

    # =========================================================
    # Step 1: Data Ingestion
    # =========================================================
    train_df, test_df = load_datasets(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH
    )

    # =========================================================
    # Step 2: Data Cleaning
    # =========================================================
    train_df = clean_data(train_df)

    # =========================================================
    # Display Results
    # =========================================================
    print("\n Cleaned Train Data Preview:")
    print(train_df.head())

    print("\n Test Data Preview (Unchanged):")
    print(test_df.head())

    print("\n Pipeline executed successfully!\n")


# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    main()