#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Generate SHAP-based feature importance plots for the Random Forest model.
"""

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs("plots/feature_plots", exist_ok=True)

# Load trained Random Forest model (not ensemble)
model = joblib.load("models/rf_model.joblib")

# Load feature data (drop label)
X = pd.read_csv("features/splits/train.csv").drop("label", axis=1)

# SHAP only supports tree-based models (RandomForest, XGBoost, etc.)
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Plot a waterfall plot for the first instance (optional)
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("plots/feature_plots/waterfall_rf_0.png", bbox_inches="tight")

# Plot and save SHAP summary bar plot
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("plots/feature_plots/shap_rf_bar.png", bbox_inches="tight")

print("SHAP plots saved to plots/feature_plots/")