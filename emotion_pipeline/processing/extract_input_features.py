#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Extracts a comprehensive feature vector from an audio file for emotion classification.
Includes MFCCs, deltas, ZCR, RMS, spectral features, GLCM and HOG from spectrogram.
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from python_speech_features import mfcc, delta

from emotion_pipeline.processing.utils.glcm_features import extract_glcm_features
from emotion_pipeline.processing.utils.hog_features import extract_hog_features
from emotion_pipeline.inference.visualize import plot_spectrogram

# ── Directory for saving spectrogram images ──
SPEC_IMG_DIR = Path("data_stt/spectrograms")
SPEC_IMG_DIR.mkdir(parents=True, exist_ok=True)


def extract_input_features(filepath):
    """
    Extracts a detailed set of features from the input WAV file.

    Parameters:
        filepath (str or Path): Path to the audio file to process.

    Returns:
        np.ndarray: A (1, N) feature array ready for emotion model prediction.

    Raises:
        ValueError: If the audio file cannot be read or is too short for analysis.
    """
    # ── Load audio file and validate ──
    result = sf.read(filepath)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(f"Could not read audio file or unexpected return from sf.read: {filepath}")
    signal, sr = result

    # ── Convert stereo to mono ──
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # ── Extract MFCCs and their derivatives ──
    mfcc_feat = mfcc(signal, samplerate=sr, numcep=13)
    mfcc_feat = (mfcc_feat - np.mean(mfcc_feat)) / np.std(mfcc_feat)  # Normalize
    mfcc_mean = mfcc_feat.mean(axis=0)
    delta_feat = delta(mfcc_feat, 2).mean(axis=0)
    delta_delta_feat = delta(delta(mfcc_feat, 2), 2).mean(axis=0)

    # ── Zero-Crossing Rate and Root Mean Square Energy ──
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    if len(signal) < 0.025 * sr:
        raise ValueError("Audio too short to extract meaningful features.")
    rms = np.sqrt(np.mean(signal ** 2))

    # ── Spectral features ──
    spec_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr).mean()
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr).mean()
    spec_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr).mean()
    spec_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr).mean()

    # ── Generate and save spectrogram image ──
    spec_path = SPEC_IMG_DIR / (Path(filepath).stem + ".png")
    plot_spectrogram(filepath, save_path=spec_path)

    # ── Extract GLCM and HOG from spectrogram ──
    glcm_feats = extract_glcm_features(spec_path)
    hog_feats = extract_hog_features(spec_path)

    # ── Combine all features into a single vector ──
    features = np.concatenate([
        mfcc_mean,
        delta_feat,
        delta_delta_feat,
        [zcr],
        [rms],
        [spec_centroid, spec_bandwidth, spec_rolloff, spec_contrast],
        glcm_feats,
        hog_feats
    ])

    print(f"Extracted features from {filepath}")
    return features.reshape(1, -1)