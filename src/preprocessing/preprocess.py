"""
=========================================================
Preprocessing Module
=========================================================

Purpose:
--------
Prepares data for modeling.

Includes:
---------
1. Time-based train-test split (for ML)
2. Time series split (for ARIMA/SARIMA)
3. Stationarity check (ADF Test)
4. Differencing for non-stationary data

=========================================================
"""

# ===============================
# Imports
# ===============================
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ===============================
# Function: split_ml_data
# ===============================
def split_ml_data(df: pd.DataFrame):
    """
    Time-based split for ML model (XGBoost)
    """

    print("\nSplitting ML dataset...")

    # Sort chronologically
    df = df.sort_values(by=['YEAR', 'MONTH'])

    # Time-based split (80%)
    split_year = df['YEAR'].quantile(0.8)

    train_ml = df[df['YEAR'] <= split_year]
    test_ml = df[df['YEAR'] > split_year]

    # Features and target
    X_train = train_ml.drop('Incident_Counts', axis=1)
    y_train = train_ml['Incident_Counts']

    X_test = test_ml.drop('Incident_Counts', axis=1)
    y_test = test_ml['Incident_Counts']

    print("ML Split completed")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ===============================
# Function: check_stationarity
# ===============================
def check_stationarity(series: pd.Series):
    """
    Performs Augmented Dickey-Fuller test
    """

    print("\nPerforming ADF Test...")

    result = adfuller(series)

    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")

    return result


# ===============================
# Function: apply_differencing
# ===============================
def apply_differencing(df: pd.DataFrame):
    """
    Applies differencing to make time series stationary
    """

    print("\nApplying differencing...")

    df['Total_Crimes_diff'] = df['Total_Crimes'].diff()

    # Drop null values created by differencing
    df_diff = df.dropna()

    print("Differencing applied")

    return df_diff


# ===============================
# Function: split_time_series
# ===============================
def split_time_series(df: pd.DataFrame):
    """
    Time-based split for ARIMA/SARIMA
    """

    print("\nSplitting time series data...")

    train_size = int(len(df) * 0.8)

    train_ts = df.iloc[:train_size]
    test_ts = df.iloc[train_size:]

    print(f"Train TS: {train_ts.shape}, Test TS: {test_ts.shape}")

    return train_ts, test_ts


# ===============================
# Full Preprocessing Pipeline
# ===============================
def preprocessing_pipeline(aggregated_df, time_series_df):
    """
    Runs full preprocessing pipeline
    """

    print("\nStarting preprocessing...\n")

    # ML Split
    X_train, X_test, y_train, y_test = split_ml_data(aggregated_df)

    # Stationarity Check
    check_stationarity(time_series_df['Total_Crimes'])

    # Differencing
    ts_diff = apply_differencing(time_series_df)

    # Time Series Split
    train_ts, test_ts = split_time_series(ts_diff)

    print("\nPreprocessing completed\n")

    return X_train, X_test, y_train, y_test, train_ts, test_ts