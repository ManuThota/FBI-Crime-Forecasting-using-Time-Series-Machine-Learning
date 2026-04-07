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
from src.feature_engineering.features import feature_engineering_pipeline
from src.config.config import TRAIN_DATA_PATH, TEST_DATA_PATH


# ===============================
# Main Function
# ===============================
def main():
    """
    Main execution pipeline
    """

    print("\nStarting FBI Crime Forecasting Pipeline...\n")

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
    # Step 3: Feature Engineering
    # =========================================================
    aggregated_df, time_series_df = feature_engineering_pipeline(train_df)

    # =========================================================
    # Display Results
    # =========================================================
    print("\nCleaned Train Data Preview:")
    print(train_df.head())

    print("\nAggregated Data Preview:")
    print(aggregated_df.head())

    print("\nTime Series Data Preview:")
    print(time_series_df.head())

    print("\nTest Data Preview (Unchanged):")
    print(test_df.head())

    print("\nPipeline executed successfully.\n")


# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    main()