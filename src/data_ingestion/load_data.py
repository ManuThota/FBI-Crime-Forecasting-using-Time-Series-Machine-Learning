"""
=========================================================
Data Ingestion Module
=========================================================

Purpose:
--------
This module is responsible for loading raw data from disk
(Train.csv and Test.csv) into pandas DataFrames.

Why this is important:
----------------------
- Separates data loading from logic
- Makes code reusable and testable
- Helps in scaling pipelines

Author: Your Name
=========================================================
"""

# ===============================
# Imports
# ===============================
import os
import pandas as pd


# ===============================
# Function: load_csv
# ===============================
def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataset

    Raises:
    -------
    FileNotFoundError:
        If file does not exist
    """

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        print(f"Successfully loaded data from: {file_path}")
        print(f"Shape: {df.shape}")

        return df

    except Exception as e:
        raise Exception(f"Error loading file: {e}")


# ===============================
# Function: load_datasets
# ===============================
def load_datasets(train_path: str, test_path: str):
    """
    Loads both train and test datasets.

    Parameters:
    -----------
    train_path : str
    test_path : str

    Returns:
    --------
    tuple
        (train_df, test_df)
    """

    print("\nLoading datasets...\n")

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    print("\n Data ingestion completed!\n")

    return train_df, test_df