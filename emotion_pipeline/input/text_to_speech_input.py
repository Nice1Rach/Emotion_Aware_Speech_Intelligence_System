#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Generate a neutral TTS waveform from input text and save as WAV.
"""

import pyttsx3
import os

SAVE_DIR = os.path.join(os.getcwd(), "Data_tts", "neutral")
WAV_PATH = os.path.join(SAVE_DIR, "neutral_tts.wav")

# ───────────────────────────────────────────────
# Function: Convert text to TTS and save as .wav
# ───────────────────────────────────────────────
def text_to_speech(text: str) -> str:
    """
    Convert the input text to speech and save as a WAV file.

    Parameters:
        text (str): Text to synthesize.

    Returns:
        str: File path to the saved WAV file.
    """
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)

        engine = pyttsx3.init()
        engine.save_to_file(text, WAV_PATH)
        engine.runAndWait()

        print(f"[TTS] Saved as WAV: {WAV_PATH}")
        return WAV_PATH

    except Exception as e:
        print(f"[TTS Error] {e}")
        return ""