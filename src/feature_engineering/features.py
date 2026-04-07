"""
=========================================================
Feature Engineering Module
=========================================================

Purpose:
--------
This module transforms cleaned data into model-ready format.

Includes:
---------
1. Aggregation for ML model (XGBoost)
2. One-Hot Encoding for categorical feature (TYPE)
3. Time Series dataset creation (for ARIMA/SARIMA)

=========================================================
"""

# ===============================
# Imports
# ===============================
import pandas as pd


# ===============================
# Function: create_aggregated_data
# ===============================
def create_aggregated_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates dataset:
    YEAR, MONTH, TYPE → Incident_Counts
    """

    print("\nCreating aggregated dataset...")

    aggregated_df = (
        df.groupby(['YEAR', 'MONTH', 'TYPE'])
        .size()
        .reset_index(name='Incident_Counts')
    )

    print("Aggregation completed")
    print(f"Shape: {aggregated_df.shape}")

    return aggregated_df


# ===============================
# Function: encode_categorical
# ===============================
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies One-Hot Encoding on TYPE column
    """

    print("\nApplying One-Hot Encoding...")

    df_encoded = pd.get_dummies(
        df,
        columns=['TYPE'],
        drop_first=True
    )

    print("Encoding completed")
    print(f"New Shape: {df_encoded.shape}")

    return df_encoded


# ===============================
# Function: create_time_series
# ===============================
def create_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time series dataset:
    Monthly total crimes
    """

    print("\nCreating time series dataset...")

    monthly_total = (
        df.groupby(['YEAR', 'MONTH'])
        .size()
        .reset_index(name='Total_Crimes')
    )

    # Create datetime column
    monthly_total['Date'] = pd.to_datetime(
        monthly_total[['YEAR', 'MONTH']].assign(DAY=1)
    )

    # Sort values
    monthly_total = monthly_total.sort_values('Date')

    # Set index
    monthly_total.set_index('Date', inplace=True)

    # IMPORTANT FIX: Set frequency explicitly
    monthly_total = monthly_total.asfreq('MS')   # MS = Month Start

    print("Time series dataset created with monthly frequency")

    return monthly_total


# ===============================
# Full Feature Pipeline
# ===============================
def feature_engineering_pipeline(df: pd.DataFrame):
    """
    Runs full feature engineering pipeline
    """

    print("\nStarting feature engineering...\n")

    # Aggregation (for ML model)
    aggregated_df = create_aggregated_data(df)

    # Encoding (for XGBoost)
    aggregated_encoded = encode_categorical(aggregated_df)

    # Time Series dataset (for ARIMA/SARIMA)
    time_series_df = create_time_series(df)

    print("\nFeature engineering completed!\n")

    return aggregated_encoded, time_series_df