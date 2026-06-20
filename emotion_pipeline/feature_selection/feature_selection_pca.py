#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Apply PCA to training features and plot explained variance curve.
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from emotion_pipeline.processing.utils.utils_train import load_and_preprocess

# ── Load preprocessed data ──
X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess()

# ── Apply PCA ──
pca = PCA(n_components=30)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# ── Plot variance explained ──
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Explained")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/embed/pca_variance_curve.png")  # Optional save
plt.show()

# Print top-30 explained variance sum
print(f"🔍 Total variance explained by 30 components: {np.sum(pca.explained_variance_ratio_):.4f}")