#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Predict emotion from an input WAV file using a trained voting ensemble model.
"""

from joblib import load
import numpy as np
import pandas as pd
import librosa
from python_speech_features import mfcc, logfbank

# ───────────────────────────────────────────────
# Load the trained voting ensemble model
# ───────────────────────────────────────────────
model = load("models/voting_ensemble_balanced.joblib")  # Optional preload

# ───────────────────────────────────────────────
# Feature Extraction Function
# ───────────────────────────────────────────────
def extract_features(audio_path: str) -> np.ndarray:
    """
    Extracts MFCC, filterbank, and ZCR features from a WAV file.

    Parameters:
        audio_path (str): Path to the audio file.

    Returns:
        np.ndarray: 1D feature vector (71 dimensions).
    """
    y, sr = librosa.load(audio_path)

    mfccs = mfcc(y, int(sr), numcep=13)
    fbank = logfbank(y, int(sr), nfilt=22)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = np.hstack([
        np.mean(mfccs, axis=0),     # 13
        np.std(mfccs, axis=0),      # 13
        np.mean(fbank, axis=0),     # 22
        np.std(fbank, axis=0),      # 22
        np.mean(zcr, axis=1)        # 1
    ])  # → Total: 71 features

    return features.reshape(1, -1)

# ───────────────────────────────────────────────
# Emotion Prediction Function
# ───────────────────────────────────────────────
def predict_emotion(filepath: str) -> tuple[str, float]:
    """
    Predict the emotion from an audio file using a pre-trained model.

    Parameters:
        filepath (str): Path to the input .wav file

    Returns:
        tuple[str, float]: Predicted emotion label and its confidence (0–1)
    """
    features = extract_features(filepath)

    # Load model and label encoder
    model = load("models/voting_ensemble_balanced.joblib")
    label_encoder = load("models/label_encoder.joblib")

    probs = model.predict_proba(features)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    pred_class = model.predict(features)[0]
    emotion = label_encoder.inverse_transform([pred_class])[0]

    return emotion, confidence