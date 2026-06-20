#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project - Feature Importance Analysis
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Compare feature importance between RF and MLP within the final Voting Ensemble.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from emotion_pipeline.processing.utils.utils_train import load_and_preprocess

# ── Load data and components ──
X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess()
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Get feature names from DataFrame if available
if isinstance(X_train, pd.DataFrame):
    feature_names = X_train.columns.tolist()
else:
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

# ── SHAP for Random Forest ──
print("Generating SHAP plot for Random Forest...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("plots/shap/shap_rf_bar_updated.png")
plt.close()

# ── Permutation Importance for MLP ──
print("Generating permutation importance plot for MLP...")
mlp_model = MLPClassifier(hidden_layer_sizes=(256,128,64), solver='lbfgs', alpha=0.001,
                          learning_rate='constant', random_state=42, max_iter=3000)
mlp_model.fit(X_train, y_train)

result = permutation_importance(mlp_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

sorted_idx = result['importances_mean'].argsort()[::-1][:15]  # top 15
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), result['importances_mean'][sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Top MLP Feature Importances")
plt.tight_layout()
plt.savefig("plots/importance/mlp_permutation_importance.png")
plt.close()

print("Comparison plots saved to 'plots/shap/' and 'plots/importance/'")
