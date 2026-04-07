"""
=========================================================
Model Evaluation Module
=========================================================

Purpose:
--------
Evaluates models and saves results to reports/

=========================================================
"""

# ===============================
# Imports
# ===============================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os


# ===============================
# Function: evaluate_regression
# ===============================
def evaluate_regression(y_true, y_pred):
    """
    Calculates evaluation metrics
    """

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2


# ===============================
# Function: save_results
# ===============================
def save_results(results, file_path="reports/results.txt"):
    """
    Saves evaluation results to file
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as f:
        for model_name, metrics in results.items():
            f.write(f"{model_name}\n")
            f.write(f"RMSE: {metrics['RMSE']}\n")
            f.write(f"MAE: {metrics['MAE']}\n")
            f.write(f"R2: {metrics['R2']}\n")
            f.write("\n")

    print(f"Results saved to {file_path}")


# ===============================
# Function: evaluate_models
# ===============================
def evaluate_models(xgb_model, arima_model, sarima_model,
                    X_test, y_test, test_ts):
    """
    Evaluates all models and saves results
    """

    print("\nStarting model evaluation...\n")

    results = {}

    # ===============================
    # XGBoost
    # ===============================
    xgb_preds = xgb_model.predict(X_test)
    rmse, mae, r2 = evaluate_regression(y_test, xgb_preds)

    print("\nXGBoost Performance:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    results["XGBoost"] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    # ===============================
    # ARIMA
    # ===============================
    arima_preds = arima_model.forecast(steps=len(test_ts))
    rmse, mae, r2 = evaluate_regression(test_ts['Total_Crimes_diff'], arima_preds)

    print("\nARIMA Performance:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    results["ARIMA"] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    # ===============================
    # SARIMA
    # ===============================
    sarima_preds = sarima_model.forecast(steps=len(test_ts))
    rmse, mae, r2 = evaluate_regression(test_ts['Total_Crimes_diff'], sarima_preds)

    print("\nSARIMA Performance:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    results["SARIMA"] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    # ===============================
    # Save Results
    # ===============================
    save_results(results)

    print("\nModel evaluation completed\n")

    return results