"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Generate emotionally-enriched assistant responses using GPT-4 based on detected emotion.
"""

"""
Audio playback utility for WAV files.
Supports playing predicted emotion clips and TTS responses.
"""

import simpleaudio as sa
import os

import simpleaudio as sa
import os
import soundfile as sf

def play_audio(file_path: str):
    """Play a WAV audio file if it's valid and non-empty."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    try:
        result = sf.read(file_path)
        if result is None or not isinstance(result, tuple) or len(result) != 2:
            print("Audio file is empty, invalid, or could not be read.")
            return
        data, sr = result
        if data is None or len(data) == 0:
            print("Audio file is empty or invalid.")
            return
    except Exception as e:
        print(f"Invalid audio file: {e}")
        return

    try:
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        print("Playback finished.")
    except Exception as e:
        print(f"Error playing audio: {e}")