#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Augments a neutral WAV file into emotion-enriched variants
(happy, sad, angry, calm, fear, surprise) using pitch, speed, volume, and
audio effects such as reverb, delay, EQ, and filters. Also saves waveform and
spectrogram visualizations for each.
"""

import os
import numpy as np
import scipy.signal as signal
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

# === Directories ===
neutral_input_path = "Data_stt/neutral/neutral_stt.wav"
emotional_output_dir = Path("Data_stt/emotional")
waveform_plot_dir = Path("plots/tts_waveforms")
spectrogram_plot_dir = Path("plots/tts_spectrograms")

emotional_output_dir.mkdir(parents=True, exist_ok=True)
waveform_plot_dir.mkdir(parents=True, exist_ok=True)
spectrogram_plot_dir.mkdir(parents=True, exist_ok=True)

# === Emotion specifications ===
base_specs = {
    "happy":    {"volume": 5,  "speed": 1.1, "pitch_shift": 2,  "vibrato": False, "tremolo": False, "reverb": False, "delay": False, "filter": None,       "eq": "treble"},
    "sad":      {"volume": -5, "speed": 0.9, "pitch_shift": -2, "vibrato": False, "tremolo": True,  "reverb": True,  "delay": True,  "filter": "lowpass",  "eq": "bass"},
    "angry":    {"volume": 8,  "speed": 1.2, "pitch_shift": 1,  "vibrato": False, "tremolo": True,  "reverb": False, "delay": False, "filter": "highpass", "eq": "treble"},
    "calm":     {"volume": 0,  "speed": 1.0, "pitch_shift": 0,  "vibrato": False, "tremolo": False, "reverb": True,  "delay": False, "filter": "bandpass", "eq": None},
    "fear":     {"volume": -2, "speed": 1.0, "pitch_shift": -1, "vibrato": True,  "tremolo": True,  "reverb": True,  "delay": False, "filter": "bandpass", "eq": "bass"},
    "surprise": {"volume": 6,  "speed": 1.3, "pitch_shift": 3,  "vibrato": False, "tremolo": False, "reverb": False, "delay": False, "filter": None,       "eq": "treble"}
}

# === Effect Functions ===
def normalize_amplitude(wav, target_dBFS=-20):
    """Normalize waveform amplitude to maintain consistent perceived volume."""
    rms = np.sqrt(np.mean(wav ** 2))
    if rms == 0:
        return wav  # prevent division by zero
    scalar = 10 ** (target_dBFS / 20) / rms
    return wav * scalar

def apply_vibrato(y, sr, rate=5, depth=0.003):
    t = np.arange(len(y)) / sr
    modulator = depth * np.sin(2 * np.pi * rate * t)
    indices = np.arange(len(y)) + modulator * sr
    indices = np.clip(indices, 0, len(y) - 1)
    return np.interp(indices, np.arange(len(y)), y)

def apply_volume(y, db):
    return y * (10 ** (db / 20))

def apply_tremolo(y, sr, rate=5):
    t = np.arange(len(y)) / sr
    return y * (0.5 * (1 + np.sin(2 * np.pi * rate * t)))

def apply_reverb(y, decay=0.3):
    reverb = np.convolve(y, np.ones(int(0.02 * 44100)) * decay, mode='full')[:len(y)]
    return y + reverb * 0.5

def apply_delay(y, sr, delay_time=0.2, decay=0.5):
    delay_samples = int(sr * delay_time)
    echo = np.zeros(len(y) + delay_samples)
    echo[:len(y)] = y
    echo[delay_samples:] += y * decay
    return echo[:len(y)]

def apply_filter(y, sr, ftype):
    y = np.asarray(y).flatten()
    nyq = sr / 2
    if ftype is None:
        return np.asarray(y)
    try:
        if ftype == "lowpass":
            result = signal.butter(4, 1000 / nyq, btype='low', output='ba')
        elif ftype == "highpass":
            result = signal.butter(4, 1000 / nyq, btype='high', output='ba')
        elif ftype == "bandpass":
            result = signal.butter(4, [500 / nyq, 3000 / nyq], btype='band', output='ba')
        else:
            return np.asarray(y)
        if result is not None and len(result) == 2:
            b, a = result
            return signal.filtfilt(b, a, y)
        else:
            return np.asarray(y)
    except Exception:
        return np.asarray(y)

def apply_eq(y, sr, eq_type):
    if eq_type == "bass":
        bass = apply_filter(y, sr, "lowpass")
        return y + 0.3 * bass
    elif eq_type == "treble":
        treble = apply_filter(y, sr, "highpass")
        return y + 0.3 * treble
    return y

def apply_emotion_effects(y, sr, params):
    y = apply_volume(y, params.get("volume", 0))

    if params.get("speed", 1.0) != 1.0:
        y = librosa.effects.time_stretch(y.astype(np.float32), rate=params["speed"])
    if params.get("pitch_shift", 0) != 0:
        y = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=params["pitch_shift"])
    if params.get("vibrato"):
        y = apply_vibrato(y, sr)
    if params.get("tremolo"):
        y = apply_tremolo(y, sr)
    if params.get("reverb"):
        y = apply_reverb(y)
    if params.get("delay"):
        y = apply_delay(y, sr)
    if params.get("filter"):
        y = apply_filter(y, sr, params["filter"])
    if params.get("eq"):
        y = apply_eq(y, sr, params["eq"])

    y = normalize_amplitude(y)  # Ensure consistent volume
    return y

# === Plotting ===
def plot_waveform(y, sr, out_path):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_spectrogram(y, sr, out_path):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# === Main Driver ===
def generate_emotions(neutral_input_path):
    y, sr = sf.read(neutral_input_path)
    base_name = os.path.splitext(os.path.basename(neutral_input_path))[0]

    for emotion, params in base_specs.items():
        y_mod = apply_emotion_effects(y, sr, params)

        wav_name = f"{base_name}_{emotion}.wav"
        wav_path = emotional_output_dir / wav_name
        sf.write(wav_path, y_mod, sr)

        wave_plot_path = waveform_plot_dir / wav_name.replace('.wav', '_wave.png')
        spec_plot_path = spectrogram_plot_dir / wav_name.replace('.wav', '_spec.png')

        plot_waveform(y_mod, sr, wave_plot_path)
        plot_spectrogram(y_mod, sr, spec_plot_path)

        print(f"[AUG] Saved {wav_path.name} with plots.")

# === Execute ===
if __name__ == "__main__":
    generate_emotions(neutral_input_path)