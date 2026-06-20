#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Generate emotionally-enriched assistant responses using GPT-4 based on detected emotion.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_emotionally_enriched_reply(original_text: str, emotion: str) -> str:
    prompt = f"""
    You are an emotionally intelligent assistant.
    
    Detected emotion: {emotion}
    
    Original user sentence:
    "{original_text}"
    
    Respond naturally as a human would.
    
    Requirements:
    - Acknowledge the emotion.
    - Reference something from the user's sentence.
    - Do not sound like a therapist.
    - Avoid generic phrases such as:
      "I'm sorry you're feeling that way"
      "That sounds difficult"
      "I understand"
    
    Instead provide a unique response.
    
    Keep the response between 2 and 4 sentences.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "[No response generated]"
    except Exception as e:
        print(f"OpenAI error: {e}")
        return f"[Fallback] {original_text}"