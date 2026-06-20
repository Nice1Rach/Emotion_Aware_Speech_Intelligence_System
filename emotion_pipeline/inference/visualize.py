#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Generate waveform, spectrogram, and pitch plots from a WAV file using matplotlib.
"""

from matplotlib.figure import Figure
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

# ───────────────────────────────────────────────
# Function: Plot waveform from audio file
# ───────────────────────────────────────────────
def plot_waveform(filepath):
    """
    Plot the waveform of an audio file.

    Parameters:
        filepath (str): Path to WAV file.

    Returns:
        matplotlib.figure.Figure: Figure object containing the waveform plot.
    """
    result = sf.read(filepath)
    if result is None:
        raise ValueError(f"Could not read audio file: {filepath}")
    signal, sr = result
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    fig = Figure(figsize=(5, 2), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, len(signal) / sr, num=len(signal)), signal)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

# ───────────────────────────────────────────────
# Function: Plot spectrogram from audio file
# ───────────────────────────────────────────────
def plot_spectrogram(filepath, save_path=None):
    """
    Plot a spectrogram of the audio signal.

    Parameters:
        filepath (str): Path to WAV file.
        save_path (str, optional): If specified, saves the figure as PNG.

    Returns:
        matplotlib.figure.Figure: Figure object containing the spectrogram.
    """
    signal, sr = sf.read(filepath)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    fig = Figure(figsize=(5, 2.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.specgram(signal, NFFT=1024, Fs=sr, noverlap=512, cmap='viridis')
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if save_path:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        canvas = FigureCanvas(fig)
        canvas.print_png(str(save_path))  # Save as PNG

    return fig

# ───────────────────────────────────────────────
# Function: Estimate pitch over time
# ───────────────────────────────────────────────
def plot_pitch(filepath):
    """
    Estimate and plot dominant pitch over time using FFT peak detection.

    Parameters:
        filepath (str): Path to WAV file.

    Returns:
        matplotlib.figure.Figure: Figure object with pitch plot.
    """
    signal, sr = sf.read(filepath)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    frame_size = 2048
    hop_size = 512
    pitches = []
    times = []

    for i in range(0, len(signal) - frame_size, hop_size):
        frame = signal[i:i + frame_size]
        spectrum = np.abs(np.fft.rfft(frame))
        peaks, _ = find_peaks(spectrum)
        if len(peaks) > 0:
            dominant_freq = peaks[np.argmax(spectrum[peaks])]
            pitch = dominant_freq * sr / frame_size
            pitches.append(pitch)
            times.append(i / sr)

    fig = Figure(figsize=(5, 2.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(times, pitches)
    ax.set_title("Estimated Pitch Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig