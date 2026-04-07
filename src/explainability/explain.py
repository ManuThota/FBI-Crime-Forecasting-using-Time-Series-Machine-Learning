"""
=========================================================
Model Explainability Module
=========================================================

Purpose:
--------
Displays feature importance from XGBoost model.

=========================================================
"""

# ===============================
# Imports
# ===============================
import pandas as pd


# ===============================
# Function: get_feature_importance
# ===============================
def get_feature_importance(model, feature_names):
    """
    Extracts and prints feature importance
    """

    print("\nCalculating feature importance...\n")

    importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    print(feature_importance_df.head(10))

    return feature_importance_df