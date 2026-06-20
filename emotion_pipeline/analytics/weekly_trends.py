#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  ID: MDS2000509
Date: 23/06/2025
Description: Visualize and display emotion history trends, including:
- Weekly emotion frequency line chart
- Emotion leaderboard bar chart
- Optional Tkinter popup leaderboard
"""

import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Toplevel, Text

# ────────────────────────────────────────────────────────
# Function: Load and process emotion history
# ────────────────────────────────────────────────────────
def load_emotion_history(emotion_history):
    """
    Convert emotion history into a DataFrame with date and timestamp columns.

    Parameters:
        emotion_history (list of tuples): (timestamp, emotion, file path)

    Returns:
        pd.DataFrame: Formatted DataFrame with timestamp and date columns
    """
    df = pd.DataFrame(emotion_history, columns=["timestamp", "emotion", "file"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df

# ────────────────────────────────────────────────────────
# Function: Plot weekly emotion trend chart
# ────────────────────────────────────────────────────────
def plot_weekly_trends(df):
    """
    Display a line chart of daily emotion frequencies.

    Parameters:
        df (pd.DataFrame): DataFrame containing emotion history with a 'date' column
    """
    weekly_counts = df.groupby(["date", "emotion"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    weekly_counts.plot(kind="line", ax=ax, marker="o", title="Weekly Emotion Trend")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────
# Function: Show emotion leaderboard as bar chart
# ────────────────────────────────────────────────────────
def show_leaderboard_chart(df):
    """
    Display a horizontal bar chart of total emotion counts.

    Parameters:
        df (pd.DataFrame): DataFrame containing an 'emotion' column
    """
    mood_ranking = df["emotion"].value_counts().reset_index()
    mood_ranking.columns = ["emotion", "count"]

    fig, ax = plt.subplots(figsize=(6, 4))
    mood_ranking.plot.barh(x='emotion', y='count', ax=ax, color='skyblue', legend=False)
    ax.set_title("🏆 Emotion Leaderboard")
    ax.set_xlabel("Frequency")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────────────────────
# Function: Optional GUI leaderboard in tkinter
# ────────────────────────────────────────────────────────
def show_leaderboard_gui(df, root=None):
    """
    Show a leaderboard of emotions in a popup Tkinter window.

    Parameters:
        df (pd.DataFrame): DataFrame with emotion column
        root (Tk, optional): Root window if integrating into existing app
    """
    mood_ranking = df["emotion"].value_counts().reset_index()
    mood_ranking.columns = ["emotion", "count"]

    popup = Toplevel(root) if root else Toplevel()
    popup.title("🏆 Emotion Leaderboard")
    popup.geometry("400x300")

    text = Text(popup, font=("Arial", 12))
    text.pack(expand=True, fill='both')

    leaderboard_text = "\n".join([f"{row.emotion.title():<10} → {row['count']}" for _, row in mood_ranking.iterrows()])
    text.insert("1.0", leaderboard_text)