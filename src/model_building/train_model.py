"""
=========================================================
Model Training Module
=========================================================

Purpose:
--------
This module handles training of:
1. XGBoost Regressor (Primary Model)
2. ARIMA Model
3. SARIMA Model

Also saves trained model to disk.

=========================================================
"""

# ===============================
# Imports
# ===============================
import joblib
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ===============================
# Function: train_xgboost
# ===============================
def train_xgboost(X_train, y_train):
    """
    Trains XGBoost Regressor
    """

    print("\nTraining XGBoost model...")

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("XGBoost training completed")

    return model


# ===============================
# Function: train_arima
# ===============================
def train_arima(train_ts):
    """
    Trains ARIMA model
    """

    print("\nTraining ARIMA model...")

    model = ARIMA(train_ts['Total_Crimes_diff'], order=(1, 1, 1))
    model_fit = model.fit()

    print("ARIMA training completed")

    return model_fit


# ===============================
# Function: train_sarima
# ===============================
def train_sarima(train_ts):
    """
    Trains SARIMA model
    """

    print("\nTraining SARIMA model...")

    model = SARIMAX(
        train_ts['Total_Crimes_diff'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    model_fit = model.fit(disp=False)

    print("SARIMA training completed")

    return model_fit


# ===============================
# Function: save_model
# ===============================
def save_model(model, path):
    """
    Saves trained model to disk
    """

    joblib.dump(model, path)

    print(f"Model saved at: {path}")


# ===============================
# Full Training Pipeline
# ===============================
def model_training_pipeline(X_train, y_train, train_ts, model_path):
    """
    Runs full model training pipeline
    """

    print("\nStarting model training...\n")

    # Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)

    # Train ARIMA
    arima_model = train_arima(train_ts)

    # Train SARIMA
    sarima_model = train_sarima(train_ts)

    # Save only best model (XGBoost)
    save_model(xgb_model, model_path)

    print("\nModel training completed\n")

    return xgb_model, arima_model, sarima_model