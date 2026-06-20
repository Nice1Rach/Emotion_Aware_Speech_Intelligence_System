"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Split scaled emotional feature dataset into stratified training and testing sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ── Paths ──
INPUT_CSV = Path("features/scaled_cleaned_features.csv")
OUTPUT_DIR = Path("features/splits")
TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

def split_dataset(test_size=0.2, random_state=42):
    """
    Split the dataset into stratified training and testing sets.
    Ensures at least one test sample per class.

    Args:
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Seed for reproducibility.
    """
    # Load dataset
    df = pd.read_csv(INPUT_CSV)

    # Separate features and labels
    X = df.drop("label", axis=1)
    y = df["label"]

    # Ensure at least 4 samples in test set (1 per class)
    test_size = max(test_size, 4 / len(df))

    # Stratified split to preserve label distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Recombine features and labels
    df_train = X_train.copy()
    df_train["label"] = y_train.values
    df_test = X_test.copy()
    df_test["label"] = y_test.values

    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(TRAIN_PATH, index=False)
    df_test.to_csv(TEST_PATH, index=False)

    print(f"Training set saved to: {TRAIN_PATH} ({len(df_train)} samples)")
    print(f"Test set saved to: {TEST_PATH} ({len(df_test)} samples)")

if __name__ == "__main__":
    split_dataset()