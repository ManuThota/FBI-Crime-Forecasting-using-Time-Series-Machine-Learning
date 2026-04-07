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
from src.model_building.evaluate import evaluate_models
from src.explainability.explain import get_feature_importance
from src.config.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH


# ===============================
# Main Function
# ===============================
def main():

    print("\nStarting FBI Crime Forecasting Pipeline...\n")

    # Step 1: Data Ingestion
    train_df, test_df = load_datasets(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH
    )

    # Step 2: Data Cleaning
    train_df = clean_data(train_df)

    # Step 3: Feature Engineering
    aggregated_df, time_series_df = feature_engineering_pipeline(train_df)

    # Step 4: Preprocessing
    X_train, X_test, y_train, y_test, train_ts, test_ts = preprocessing_pipeline(
        aggregated_df,
        time_series_df
    )

    # Step 5: Model Training
    xgb_model, arima_model, sarima_model = model_training_pipeline(
        X_train,
        y_train,
        train_ts,
        MODEL_SAVE_PATH
    )

    # Step 6: Evaluation
    evaluate_models(
        xgb_model,
        arima_model,
        sarima_model,
        X_test,
        y_test,
        test_ts
    )

    # Step 7: Explainability
    feature_importance = get_feature_importance(
        xgb_model,
        X_train.columns
    )

    print("\nPipeline executed successfully.\n")


# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    main()