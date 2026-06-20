#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Segments emotional WAV audio into 1-second clips with 50% overlap,
saves the clips, and generates preview plots (waveforms and spectrograms).
"""

import os
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import librosa.display

# ── CONFIGURATION ───────────────────────────────
WIN_SEC = 1.0       # Segment window length (seconds)
HOP_SEC = 0.5       # Hop length (seconds)
FIGSIZE = (8, 2)    # Plot dimensions
DPI = 60            # Plot resolution

# ── Default Paths ───────────────────────────────
emotional_input_dir = Path("Data_stt/emotional")
SEGMENTS_DIR = "data_stt/segment_clips"
WAVE_PLOTS = "plots/tts_segment_waveforms"
SPECT_PLOTS = "plots/tts_spectrograms"


def run_segmentation(emotional_input_dir: str, segments_dir: str = SEGMENTS_DIR) -> None:
    """
    Segments WAV files into 1-second windows with 0.5s hop.
    Saves audio segments, waveform previews, and mel spectrogram plots.

    Args:
        emotional_input_dir (str): Directory containing input WAV files.
        segments_dir (str): Directory to store segmented audio clips.
    """
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(WAVE_PLOTS, exist_ok=True)
    os.makedirs(SPECT_PLOTS, exist_ok=True)

    # Find all WAV files recursively
    pattern = os.path.join(emotional_input_dir, "**", "*.wav")
    all_wavs = glob(pattern, recursive=True)

    if not all_wavs:
        print(f"No WAV files found under {emotional_input_dir}")
        return

    for wav_path in all_wavs:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        y, sr = librosa.load(wav_path, sr=None)
        win_len = int(WIN_SEC * sr)
        hop_len = int(HOP_SEC * sr)

        segments = []
        # Segment signal with sliding window
        for start in range(0, len(y) - win_len + 1, hop_len):
            seg = y[start:start + win_len]
            timestamp = start / sr
            out_name = f"{stem}_t{timestamp:.2f}s.wav"
            out_path = os.path.join(segments_dir, out_name)
            sf.write(out_path, seg, sr)
            segments.append(out_path)

        # Preview first 4 segments as waveform and spectrograms
        if len(segments) >= 4:
            # Plot waveform previews
            fig, axes = plt.subplots(1, 4, figsize=FIGSIZE)
            for i, ax in enumerate(axes):
                y_seg, _ = librosa.load(segments[i], sr=None)
                ax.plot(y_seg, linewidth=0.5)
                ax.set_title(f"{i * HOP_SEC:.1f}s", fontsize=6)
                ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(os.path.join(WAVE_PLOTS, f"{stem}_wave.png"), dpi=DPI, bbox_inches="tight")
            plt.close(fig)

            # Plot mel spectrogram previews
            fig, axes = plt.subplots(1, 4, figsize=FIGSIZE)
            for i, ax in enumerate(axes):
                y_seg, sr_seg = librosa.load(segments[i], sr=None)
                S = librosa.feature.melspectrogram(y=y_seg, sr=sr_seg)
                S_db = librosa.power_to_db(S, ref=np.max)
                librosa.display.specshow(S_db, sr=sr_seg, x_axis='time', y_axis='mel', ax=ax)
                ax.set_title(f"{i}", fontsize=6)
                ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(os.path.join(SPECT_PLOTS, f"{stem}_spect.png"), dpi=DPI, bbox_inches="tight")
            plt.close(fig)

    print(f"Segmentation complete.\n"
          f"Clips saved in:         {segments_dir}\n"
          f"Waveform plots in:     {WAVE_PLOTS}\n"
          f"Spectrograms in:       {SPECT_PLOTS}")


def main():
    """
    CLI entry point to run segmentation from specified input directory.
    """
    parser = argparse.ArgumentParser(
        description="Segment WAVs into fixed-length clips with waveform/spectrogram previews."
    )
    parser.add_argument(
        "emotional_input_dir",
        help="Path to emotion-augmented WAVs (e.g., STT or TTS output folder)"
    )
    parser.add_argument(
        "--segments_dir",
        default=SEGMENTS_DIR,
        help="Where to save the segmented audio clips"
    )
    args = parser.parse_args()

    run_segmentation(args.emotional_input_dir, segments_dir=args.segments_dir)


if __name__ == "__main__":
    main()