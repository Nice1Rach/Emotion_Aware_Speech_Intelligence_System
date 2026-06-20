#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Extracts MFCC, ZCR, and RMS features from segmented speech clips, labels them by emotion 
from filenames, saves to CSV, and generates diagnostic plots.
"""

import os
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from python_speech_features import mfcc
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ── Paths ───────────────────────────────────────────────
INPUT_DIR = Path("data_stt/segment_clips")
OUTPUT_CSV = Path("features/extracted_features.csv")
PLOT_DIR = Path("plots/feature_plots")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Label Extraction from Filename ──────────────────────
def get_emotion_from_filename(filename):
    emotions = ["happy", "sad", "angry", "fear", "calm", "neutral", "surprise"]
    for emotion in emotions:
        if emotion in filename.lower():
            return emotion
    return "unknown"

# ── Feature Extraction ──────────────────────────────────
def extract_features_from_file(file_path):
    signal, sr = sf.read(file_path)

    # Ensure mono
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # MFCC (13)
    mfcc_feat = mfcc(signal, sr, numcep=13)
    mfcc_mean = mfcc_feat.mean(axis=0)

    # ZCR
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

    # RMS
    rms = np.sqrt(np.mean(signal**2))

    return np.concatenate([mfcc_mean, [zcr], [rms]])

# ── Main Feature Processing ─────────────────────────────
def process_all():
    data = []
    all_files = list(INPUT_DIR.glob("**/*.wav"))  # Recursive search

    for file in all_files:
        label = get_emotion_from_filename(file.name)
        if label == "unknown":
            print(f"Skipping file with unknown label: {file.name}")
            continue

        try:
            features = extract_features_from_file(file)
            data.append(np.append(features, label))
        except Exception as e:
            print(f"Failed to extract from {file.name}: {e}")

    # Create DataFrame
    columns = [f"mfcc_{i+1}" for i in range(13)] + ["zcr", "rms", "label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Feature extraction complete → {OUTPUT_CSV}")
    print("Label distribution:", Counter(df["label"]))

# ── Optional: Plot Features ─────────────────────────────
def generate_plots():
    df = pd.read_csv(OUTPUT_CSV)

    # 1. Boxplot for MFCC_1
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="label", y="mfcc_1")
    plt.title("Distribution of MFCC_1 by Emotion")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "boxplot_mfcc1.png")
    plt.close()

    # 2. ZCR vs RMS Scatterplot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="zcr", y="rms", hue="label")
    plt.title("ZCR vs RMS Colored by Emotion")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "zcr_rms_scatter.png")
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df.drop(columns=["label"]).astype(float).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "correlation_heatmap.png")
    plt.close()

    print(f"Plots saved in {PLOT_DIR}")

# ── Run ─────────────────────────────────────────────────
if __name__ == "__main__":
    process_all()
    generate_plots()