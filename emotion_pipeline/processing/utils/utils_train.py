"""
Media Design School - NLP303: Assignment 3  
Emotionally-Sensitive Agent Project  
Author: Rachel Heke  Id: MDS2000509  
Date: 23/06/2025  
Description: Load features, apply label encoding and scaling, then split into train/test sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load

def load_and_preprocess(path='features/scaled_cleaned_features.csv'):
    """
    Load preprocessed features, apply label encoding and feature scaling, 
    then split into training and testing datasets.

    Args:
        path (str): Path to the cleaned and scaled features CSV file.

    Returns:
        X_train (DataFrame): Training feature matrix.
        X_test (DataFrame): Testing feature matrix.
        y_train (ndarray): Training labels.
        y_test (ndarray): Testing labels.
        rf_scaler (StandardScaler): Fitted scaler object.
        rf_encoder (LabelEncoder): Fitted label encoder object.
    """
    # Load feature dataset
    df = pd.read_csv(path)

    # Split features and labels
    X = df.iloc[:, :-1]       # ✅ Keep as DataFrame
    y = df.iloc[:, -1].values

    # Load saved label encoder and scaler
    rf_encoder = load("models/label_encoder.joblib")
    rf_scaler = load("models/rf_scaler.joblib")

    # Apply label encoding and scaling
    y_encoded = rf_encoder.transform(y)
    X_scaled = rf_scaler.transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    return X_train, X_test, y_train, y_test, rf_scaler, rf_encoder