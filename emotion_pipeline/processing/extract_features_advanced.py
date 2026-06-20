"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Extracts advanced features (MFCC, delta, spectral, GLCM, HOG) from segmented emotional speech clips and saves them into a structured CSV dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from python_speech_features import mfcc

from emotion_pipeline.processing.utils.glcm_features import extract_glcm_features
from emotion_pipeline.processing.utils.hog_features import extract_hog_features
from emotion_pipeline.inference.visualize import plot_spectrogram

# ── Paths ──
AUDIO_DIR = Path("data_stt/segment_clips")
SPEC_IMG_DIR = Path("data_stt/spectrograms")
FEATURE_CSV = Path("features/combined_enhanced_features.csv")
SPEC_IMG_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── Label Extraction ──
def get_emotion_from_filename(filename):
    emotions = ["happy", "sad", "angry", "fear", "calm", "neutral", "surprise"]
    for emotion in emotions:
        if emotion in filename.lower():
            return emotion
    return "unknown"

# ── Audio Feature Extraction ──
def extract_audio_features(filepath):
    result = sf.read(filepath)
    if result is None or not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(f"Could not read audio file: {filepath}")
    signal, sr = result
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    y = signal

    # MFCCs and delta
    mfcc_feat = mfcc(signal, sr, numcep=13)
    mfcc_mean = mfcc_feat.mean(axis=0)
    delta_mfcc = librosa.feature.delta(mfcc_feat).mean(axis=0)
    delta2_mfcc = librosa.feature.delta(mfcc_feat, order=2).mean(axis=0)

    # Spectral features from librosa
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()

    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    rms = np.sqrt(np.mean(signal**2))

    return (mfcc_mean, delta_mfcc, delta2_mfcc, zcr, rms,
            spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_contrast)

# ── Main Feature Processing ──
def process_all():
    data = []
    files = list(AUDIO_DIR.glob("*.wav"))

    for file in tqdm(files, desc="Processing Audio Files"):
        label = get_emotion_from_filename(file.name)
        if label == "unknown":
            print(f"Skipping unknown label: {file.name}")
            continue

        try:
            # 1. Extract Audio Features
            (mfcc_feats, delta_mfcc, delta2_mfcc, zcr, rms,
             centroid, bandwidth, rolloff, contrast) = extract_audio_features(file)

            # Construct spectrogram image path
            spec_path = SPEC_IMG_DIR / file.with_suffix('.png').name
            plot_spectrogram(file, save_path=spec_path)

            # 2. Extract GLCM features
            glcm_feats = extract_glcm_features(spec_path)

            # 3. Extract HOG features
            hog_feats = extract_hog_features(spec_path)

            # 4. Combine all features
            combined = np.concatenate([
                mfcc_feats, delta_mfcc, delta2_mfcc,
                [zcr], [rms],
                [centroid, bandwidth, rolloff, contrast],
                glcm_feats, hog_feats, [label]
            ])
            data.append(combined)

        except Exception as e:
            print(f"Failed for {file.name}: {e}")

    # Define column names
    mfcc_cols = [f"mfcc_{i+1}" for i in range(13)]
    delta_cols = [f"delta_mfcc_{i+1}" for i in range(13)]
    delta2_cols = [f"delta2_mfcc_{i+1}" for i in range(13)]
    spectral_cols = ["zcr", "rms", "centroid", "bandwidth", "rolloff", "contrast"]

    if data:
        glcm_len = len(data[0][len(mfcc_cols + delta_cols + delta2_cols + spectral_cols):-1]) - 9  # remove hog count guess
        hog_len = len(data[0]) - (len(mfcc_cols + delta_cols + delta2_cols + spectral_cols) + glcm_len + 1)
    else:
        glcm_len = 0
        hog_len = 0

    glcm_cols = [f"glcm_{i+1}" for i in range(glcm_len)]
    hog_cols = [f"hog_{i+1}" for i in range(hog_len)]
    columns = mfcc_cols + delta_cols + delta2_cols + spectral_cols + glcm_cols + hog_cols + ["label"]

    # Save to CSV
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(FEATURE_CSV, index=False)
    print(f"Enhanced features saved to {FEATURE_CSV}")

if __name__ == "__main__":
    process_all()