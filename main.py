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
from src.preprocessing.preprocess import preprocessing_pipeline
from src.model_building.train_model import model_training_pipeline
from src.config.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH


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
    # Step 4: Preprocessing
    # =========================================================
    X_train, X_test, y_train, y_test, train_ts, test_ts = preprocessing_pipeline(
        aggregated_df,
        time_series_df
    )

    # =========================================================
    # Step 5: Model Training
    # =========================================================
    xgb_model, arima_model, sarima_model = model_training_pipeline(
        X_train,
        y_train,
        train_ts,
        MODEL_SAVE_PATH
    )

    # =========================================================
    # Display Summary
    # =========================================================
    print("\nCleaned Train Data Preview:")
    print(train_df.head())

    print("\nAggregated Data Preview:")
    print(aggregated_df.head())

    print("\nTime Series Data Preview:")
    print(time_series_df.head())

    print("\nML Training Data Shape:", X_train.shape)
    print("ML Testing Data Shape:", X_test.shape)

    print("\nTime Series Train Shape:", train_ts.shape)
    print("Time Series Test Shape:", test_ts.shape)

    print("\nTest Data Preview (Unchanged):")
    print(test_df.head())

    print("\nPipeline executed successfully.\n")


# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    main()