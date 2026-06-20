#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Generate emotionally-enriched assistant responses using GPT-4 based on detected emotion.
"""

import pyttsx3

VOICE_STYLE = {
    "angry": {"voice": "David", "rate": 205, "volume": 1.0},
    "happy": {"voice": "Zira", "rate": 200, "volume": 1.0},
    "sad": {"voice": "Zira", "rate": 140, "volume": 0.75},
    "calm": {"voice": "Zira", "rate": 150, "volume": 0.7},
    "fear": {"voice": "David", "rate": 175, "volume": 0.85},
    "surprise": {"voice": "Zira", "rate": 215, "volume": 1.0},
    "neutral": {"voice": "David", "rate": 170, "volume": 0.9},
}

def speak_emotionally(text, emotion="neutral"):
    engine = pyttsx3.init(driverName="sapi5")
    voices = engine.getProperty("voices")

    style = VOICE_STYLE.get(emotion.lower(), VOICE_STYLE["neutral"])
    selected_voice_name = style["voice"]

    for voice in voices:
        if selected_voice_name.lower() in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break

    engine.setProperty("rate", style["rate"])
    engine.setProperty("volume", style["volume"])

    print(f"Using voice: {selected_voice_name} | Rate: {style['rate']} | Volume: {style['volume']}")

    engine.say(text)
    engine.runAndWait()