"""
=========================================================
Data Cleaning Module
=========================================================

Purpose:
--------
Handles all data preprocessing steps:
1. Remove duplicate rows
2. Handle missing values:
    - Categorical → "unknown"
    - Numerical → median
3. Fix Date column:
    - Convert to datetime
    - Reconstruct missing dates
    - Fill remaining with mode

This module ensures clean, consistent data for modeling.

=========================================================
"""

# ===============================
# Imports
# ===============================
import pandas as pd


# ===============================
# Function: remove_duplicates
# ===============================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from dataset.
    """

    initial_shape = df.shape

    df = df.drop_duplicates()

    print(f"Removed duplicates: {initial_shape[0] - df.shape[0]} rows")

    return df


# ===============================
# Function: handle_missing_values
# ===============================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values:
    - Categorical → 'unknown'
    - Numerical → median
    """

    print("\n Handling missing values...")

    # Separate column types
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Fill categorical columns
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")

    # Fill numerical columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    print("Missing values handled")

    return df


# ===============================
# Function: fix_date_column
# ===============================
def fix_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes Date column using your notebook logic:
    1. Convert to datetime
    2. Reconstruct missing values using YEAR, MONTH, DAY
    3. Fill remaining with mode
    """

    print("\n Fixing Date column...")

    # Convert to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Reconstruct missing dates
    df['Date'] = df['Date'].fillna(
        pd.to_datetime(
            df[['YEAR', 'MONTH', 'DAY']].astype(int),
            errors='coerce'
        )
    )

    # Fill remaining missing with mode
    df['Date'] = df['Date'].fillna(df['Date'].mode()[0])

    print("Date column fixed")

    return df


# ===============================
# Main Cleaning Function
# ===============================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.
    """

    print("\n Starting data cleaning...\n")

    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = fix_date_column(df)

    print("\n Data cleaning completed!\n")

    return df