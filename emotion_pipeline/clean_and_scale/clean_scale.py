#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Generate emotionally-enriched assistant responses using GPT-4 based on detected emotion.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump
from pathlib import Path

# ─────────────────────────────
# File paths
# ─────────────────────────────
INPUT_CSV = Path("features/combined_advanced_features.csv")
OUTPUT_CSV = Path("features/scaled_cleaned_features.csv")
SCALER_PATH = Path("models/rf_scaler.joblib")

def clean_and_scale():
    """
    Clean, scale, and visualize extracted audio features.
    Saves the scaled features and fitted scaler object.
    Also generates plots of feature distributions and PCA projection.
    """
    # Load and prepare data
    df = pd.read_csv(INPUT_CSV)

    # Ensure label is last column
    if df.columns[-1] != "label":
        label_col = df.pop("label")
        df["label"] = label_col

    # Clean up infinite/NaN values
    df_clean = df.replace([float("inf"), float("-inf")], pd.NA).dropna()

    # Split features and labels
    X = df_clean.drop("label", axis=1)
    y = df_clean["label"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaled features and labels
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled["label"] = y.reset_index(drop=True)
    df_scaled.to_csv(OUTPUT_CSV, index=False)

    # Save the fitted scaler
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(scaler, SCALER_PATH)

    print(f"Scaled/cleaned features saved to {OUTPUT_CSV}")
    print(f"Scaler saved to {SCALER_PATH}")

    # ─────────────────────────────
    # Visualization
    # ─────────────────────────────
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOT_DIR = Path("plots/scaled_features")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Plot feature distributions
    for feature in ["mfcc_1", "zcr", "rms"]:
        if feature in df_scaled.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(data=df_scaled, x=feature, bins=20, kde=True)
            plt.title(f"Distribution of Scaled {feature}")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"{feature}_scaled_dist.png")
            plt.close()

    # PCA 2D projection for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["label"] = y.reset_index(drop=True)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", palette="tab10")
    plt.title("PCA Projection of Scaled Features")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pca_projection.png")
    plt.close()

    print(f"Plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    clean_and_scale()