"""
=========================================================
Main Entry Point of Project
=========================================================
"""

# ===============================
# Imports
# ===============================
from src.data_ingestion.load_data import load_datasets
from src.data_ingestion.save_data import save_all_datasets
from src.data_cleaning.cleaning import clean_data
from src.feature_engineering.features import feature_engineering_pipeline
from src.preprocessing.preprocess import preprocessing_pipeline
from src.model_building.train_model import model_training_pipeline
from src.model_building.evaluate import evaluate_models
from src.explainability.explain import get_feature_importance

from src.config.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH,
    AGGREGATED_DATA_PATH, TIME_SERIES_DATA_PATH,
    X_TRAIN_PATH, X_TEST_PATH,
    Y_TRAIN_PATH, Y_TEST_PATH,
    TS_TRAIN_PATH, TS_TEST_PATH
)


# ===============================
# Main Function
# ===============================
def main():

    print("\nStarting FBI Crime Forecasting Pipeline...\n")

    # Step 1: Load Data
    train_df, test_df = load_datasets(
        TRAIN_DATA_PATH,
        TEST_DATA_PATH
    )

    # Step 2: Clean Data
    train_df = clean_data(train_df)

    # Step 3: Feature Engineering
    aggregated_df, time_series_df = feature_engineering_pipeline(train_df)

    # Step 4: Preprocessing
    X_train, X_test, y_train, y_test, train_ts, test_ts = preprocessing_pipeline(
        aggregated_df,
        time_series_df
    )

    # Step 5: Save Processed Data
    save_all_datasets(
        aggregated_df,
        time_series_df,
        X_train, X_test,
        y_train, y_test,
        train_ts, test_ts,
        {
            "aggregated": AGGREGATED_DATA_PATH,
            "time_series": TIME_SERIES_DATA_PATH,
            "X_train": X_TRAIN_PATH,
            "X_test": X_TEST_PATH,
            "y_train": Y_TRAIN_PATH,
            "y_test": Y_TEST_PATH,
            "train_ts": TS_TRAIN_PATH,
            "test_ts": TS_TEST_PATH
        }
    )

    # Step 6: Model Training
    xgb_model, arima_model, sarima_model = model_training_pipeline(
        X_train,
        y_train,
        train_ts,
        MODEL_SAVE_PATH
    )

    # Step 7: Evaluation
    evaluate_models(
        xgb_model,
        arima_model,
        sarima_model,
        X_test,
        y_test,
        test_ts
    )

    # Step 8: Explainability
    get_feature_importance(
        xgb_model,
        X_train.columns
    )

    print("\nPipeline executed successfully.\n")


# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    main()