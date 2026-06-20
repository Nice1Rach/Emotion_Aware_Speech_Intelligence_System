#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Record microphone input, transcribe using speech recognition,
             and save the neutral voice input as a WAV file.
"""

import os
from pathlib import Path
import speech_recognition as sr
import soundfile as sf  # For WAV compatibility (not used directly but included)
import numpy as np      # For potential future preprocessing

# ─────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────
TRANSCRIPTION_DIR = Path("data_stt/transcriptions")
NEUTRAL_AUDIO_DIR = Path("data_stt/neutral")
TRANSCRIPTION_DIR.mkdir(parents=True, exist_ok=True)
NEUTRAL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────
# Main Function: Record, Transcribe, and Save Audio
# ─────────────────────────────────────────────────────
def record_and_transcribe_and_save():
    """
    Records audio from the microphone, transcribes the speech,
    and saves the waveform to a neutral audio directory.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Speak your sentence (max 5 sec)...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            if not isinstance(audio, sr.AudioData):
                print("Failed to capture audio data.")
                return None
    except Exception as mic_error:
        print(f"Microphone error: {mic_error}")
        return None

    try:
        print("Transcribing...")
        # type: ignore is used to suppress IDE/linter warning about recognize_google
        text = recognizer.recognize_google(audio)  # type: ignore[attr-defined]
        print(f"Transcription: {text}")

        # Save the WAV data to file
        wav_data = audio.get_wav_data()
        with open(NEUTRAL_AUDIO_DIR / "neutral_stt.wav", "wb") as wav_file:
            wav_file.write(wav_data)

        print("Your natural voice has been saved as neutral_stt.wav")
        return text

    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"API error: {e}")
        return None
