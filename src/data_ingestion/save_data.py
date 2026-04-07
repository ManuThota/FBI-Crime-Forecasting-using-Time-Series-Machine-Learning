"""
=========================================================
Data Saving Module
=========================================================

Purpose:
--------
Saves processed datasets into data/processed/

=========================================================
"""

# ===============================
# Imports
# ===============================
import os
import pandas as pd


# ===============================
# Function: save_dataframe
# ===============================
def save_dataframe(df: pd.DataFrame, file_path: str):
    """
    Saves DataFrame as CSV
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    df.to_csv(file_path, index=True)

    print(f"Saved: {file_path}")


# ===============================
# Function: save_all_datasets
# ===============================
def save_all_datasets(
    aggregated_df,
    time_series_df,
    X_train, X_test,
    y_train, y_test,
    train_ts, test_ts,
    paths
):
    """
    Saves all processed datasets
    """

    print("\nSaving processed datasets...\n")

    save_dataframe(aggregated_df, paths["aggregated"])
    save_dataframe(time_series_df, paths["time_series"])

    save_dataframe(X_train, paths["X_train"])
    save_dataframe(X_test, paths["X_test"])

    save_dataframe(y_train.to_frame(), paths["y_train"])
    save_dataframe(y_test.to_frame(), paths["y_test"])

    save_dataframe(train_ts, paths["train_ts"])
    save_dataframe(test_ts, paths["test_ts"])

    print("\nAll processed datasets saved successfully\n")