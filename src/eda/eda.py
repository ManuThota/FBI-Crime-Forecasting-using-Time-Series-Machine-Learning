"""
=========================================================
Exploratory Data Analysis (EDA) Module
=========================================================

Purpose:
--------
Generates visual insights from the dataset and saves plots.

Plots Included:
---------------
1. Crime Trend Over Time
2. Crime Type Distribution
3. Top Crime Types
4. Monthly Crime Pattern

=========================================================
"""

# ===============================
# Imports
# ===============================
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# Helper Function: Save Plot
# ===============================
def save_plot(fig, file_path):
    """
    Saves plot to reports/figures/
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fig.savefig(file_path)
    plt.close(fig)

    print(f"Plot saved: {file_path}")


# ===============================
# Plot 1: Crime Trend Over Time
# ===============================
def plot_crime_trend(time_series_df, save_path):
    """
    Line plot of total crimes over time
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(time_series_df.index, time_series_df['Total_Crimes'])
    ax.set_title("Crime Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Crimes")

    save_plot(fig, save_path)


# ===============================
# Plot 2: Crime Type Distribution
# ===============================
def plot_crime_type_distribution(df, save_path):
    """
    Bar plot of crime type counts
    """

    fig, ax = plt.subplots(figsize=(12, 6))

    df['TYPE'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Crime Type Distribution")
    ax.set_xlabel("Crime Type")
    ax.set_ylabel("Count")

    save_plot(fig, save_path)


# ===============================
# Plot 3: Top Crime Types
# ===============================
def plot_top_crimes(df, save_path, top_n=10):
    """
    Top N crime types
    """

    top_crimes = df['TYPE'].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))

    top_crimes.plot(kind='bar', ax=ax)
    ax.set_title(f"Top {top_n} Crime Types")
    ax.set_xlabel("Crime Type")
    ax.set_ylabel("Count")

    save_plot(fig, save_path)


# ===============================
# Plot 4: Monthly Pattern
# ===============================
def plot_monthly_pattern(df, save_path):
    """
    Monthly crime distribution
    """

    monthly_counts = df.groupby('MONTH').size()

    fig, ax = plt.subplots(figsize=(8, 5))

    monthly_counts.plot(kind='bar', ax=ax)
    ax.set_title("Monthly Crime Pattern")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Crimes")

    save_plot(fig, save_path)


# ===============================
# Full EDA Pipeline
# ===============================
def run_eda(df, time_series_df):
    """
    Runs all EDA functions
    """

    print("\nStarting EDA...\n")

    plot_crime_trend(
        time_series_df,
        "reports/figures/crime_trend.png"
    )

    plot_crime_type_distribution(
        df,
        "reports/figures/crime_type_distribution.png"
    )

    plot_top_crimes(
        df,
        "reports/figures/top_crimes.png"
    )

    plot_monthly_pattern(
        df,
        "reports/figures/monthly_pattern.png"
    )

    print("\nEDA completed. Plots saved in reports/figures/\n")