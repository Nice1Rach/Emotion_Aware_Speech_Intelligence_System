#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Visualize scaled audio features using PCA for emotion classification.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import seaborn as sns

def run_embedding():
    """
    Load cleaned and scaled feature data, apply PCA,
    and generate visualizations of the 2D projection and explained variance.
    """
    # ─────────────── File paths ───────────────
    INPUT_CSV = Path("features/cleaned_scaled_features.csv")
    PLOT_DIR = Path("plots/embed")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────── Load feature data ───────────────
    df = pd.read_csv(INPUT_CSV)
    X = df.drop("label", axis=1)
    y = df["label"]

    # ─────────────── PCA Transformation ───────────────
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    df_pca["label"] = y

    # ─────────────── Plot: PCA Projection ───────────────
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="label", palette="Set2", s=80, edgecolor="black")
    plt.title("PCA Projection of Scaled Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Emotion")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pca_projection.png")
    plt.close()

    # ─────────────── Plot: Variance Explained ───────────────
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
    plt.title("Explained Variance by PCA Components")
    plt.xlabel("PCA Component")
    plt.ylabel("Variance Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pca_variance.png")
    plt.close()

    print(f"PCA plots saved to {PLOT_DIR}")

# ─────────────── Entry point ───────────────
if __name__ == "__main__":
    run_embedding()