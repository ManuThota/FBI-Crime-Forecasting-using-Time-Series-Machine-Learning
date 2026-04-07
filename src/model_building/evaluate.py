"""
=========================================================
Model Evaluation Module
=========================================================

Purpose:
--------
Evaluates model performance using:
- RMSE
- MAE
- R2 Score

Also compares models and prints results.

=========================================================
"""

# ===============================
# Imports
# ===============================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# ===============================
# Function: evaluate_regression
# ===============================
def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    Calculates evaluation metrics
    """

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

    return rmse, mae, r2


# ===============================
# Function: evaluate_models
# ===============================
def evaluate_models(xgb_model, arima_model, sarima_model,
                    X_test, y_test, test_ts):
    """
    Evaluates all models
    """

    print("\nStarting model evaluation...\n")

    # ===============================
    # XGBoost Evaluation
    # ===============================
    xgb_preds = xgb_model.predict(X_test)
    evaluate_regression(y_test, xgb_preds, "XGBoost")

    # ===============================
    # ARIMA Evaluation
    # ===============================
    arima_preds = arima_model.forecast(steps=len(test_ts))
    evaluate_regression(test_ts['Total_Crimes_diff'], arima_preds, "ARIMA")

    # ===============================
    # SARIMA Evaluation
    # ===============================
    sarima_preds = sarima_model.forecast(steps=len(test_ts))
    evaluate_regression(test_ts['Total_Crimes_diff'], sarima_preds, "SARIMA")

    print("\nModel evaluation completed\n")