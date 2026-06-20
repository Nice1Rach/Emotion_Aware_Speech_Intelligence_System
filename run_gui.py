#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description:
Main GUI for interacting with the Emotionally-Sensitive Agent.
Includes voice recording, file upload, emotion prediction, ChatGPT response,
visualizations (waveform, spectrogram, pitch), and leaderboard analysis.
"""

import os
import queue
import threading
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from ttkbootstrap import Window
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from collections import Counter
import openai

# ======== Custom Modules ========
from emotion_pipeline.inference.predict_emotion import predict_emotion
from emotion_pipeline.inference.visualize import plot_waveform, plot_spectrogram, plot_pitch
from emotion_pipeline.audio_utils.playback import play_audio
from emotion_pipeline.audio_utils.tts_emotion_response import speak_emotionally
from emotion_pipeline.agent.agent_response import generate_emotionally_enriched_reply
from emotion_pipeline.gui.utils.plot_embedder import embed_plot

# ======== Environment Setup ========
openai.api_key = os.getenv("OPENAI_API_KEY")

# ======== Globals ========
recording = False
record_queue = queue.Queue()
emotion_history = []
uploaded_file_path = None
recorded_file_path = None
current_emotion = None

# ======== GUI Setup ========
app = Window(themename="solar")
app.title("🎵 Emotion-Aware Speech Intelligence System")
app.geometry("1280x800")

# ======== Layout Frames ========
frame_top = ttk.Frame(app)
frame_top.pack(pady=10)

frame_bottom = ttk.Frame(app)
frame_bottom.pack(fill='both', expand=True)

frame_wave = ttk.Frame(frame_bottom)
frame_wave.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

frame_spec = ttk.Frame(frame_bottom)
frame_spec.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

frame_pitch = ttk.Frame(frame_bottom)
frame_pitch.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

# ======== File State ========
current_file_path = None

# ======== Core Functions ========

def upload_audio():
    """Let user select a WAV file and plot its visualizations."""
    global uploaded_file_path
    path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if path:
        uploaded_file_path = path
        plot_all(path)

def plot_all(filepath):
    """Plot waveform, spectrogram, and pitch into separate GUI frames."""
    for frame in [frame_wave, frame_spec, frame_pitch]:
        for widget in frame.winfo_children():
            widget.destroy()
    embed_plot(plot_waveform(filepath), frame_wave)
    embed_plot(plot_spectrogram(filepath), frame_spec)
    embed_plot(plot_pitch(filepath), frame_pitch)

def run_prediction():
    global current_emotion

    filepath = uploaded_file_path or recorded_file_path
    if not filepath:
        messagebox.showwarning("No file", "Please select or record an audio file.")
        return

    try:
        predicted_emotion, confidence = predict_emotion(filepath)
        current_emotion = predicted_emotion

        label_prediction.config(text=f"Emotion: {predicted_emotion}")
        result_label.config(
            text=f"Predicted Emotion: {predicted_emotion} ({confidence * 100:.2f}%)"
        )

        emotion_history.append((
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            predicted_emotion,
            filepath
        ))
        update_history()

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

def record_audio_wrapper():
    global recorded_file_path
    filepath = record_audio()

    if filepath:
        recorded_file_path = filepath
        app.after(0, lambda: plot_all(filepath))
        app.after(0, lambda: messagebox.showinfo("Recording Saved", "Microphone recording saved. Click Predict Emotion to analyse it."))


def update_history():
    """Refresh emotion memory list display."""
    box_history.delete(0, tk.END)
    for entry in reversed(emotion_history):
        box_history.insert(tk.END, f"{entry[0]} → {entry[1]}")

def chat_response():
    """Generate and speak a personalized agent reply based on user sentence and predicted emotion."""
    if not current_emotion:
        messagebox.showwarning("Missing", "Run emotion prediction first.")
        return

    original_text = simpledialog.askstring("Original Input", "Enter the original sentence:")
    if original_text is None:
        return

    def run_enriched_reply():
        try:
            enriched_reply = generate_emotionally_enriched_reply(original_text, current_emotion)

            def update_ui():
                text_response.delete("1.0", tk.END)
                text_response.insert(tk.END, enriched_reply)
                #reply_label.config(text=f"ChatGPT Response:\n{enriched_reply}")

            app.after(0, update_ui)
            speak_emotionally(enriched_reply, current_emotion)

        except Exception as e:
            app.after(0, lambda: messagebox.showerror("Reply Error", str(e)))

    threading.Thread(target=run_enriched_reply, daemon=True).start()

def open_sandbox():
    """Sandbox tool to compare model vs manual emotion label."""
    path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if path:
        pred = predict_emotion(path)[0]
        manual = simpledialog.askstring("Manual Label", "What is the correct emotion?")
        plot_waveform(path)
        messagebox.showinfo("Comparison", f"Predicted: {pred}\nManual: {manual}")

def record_audio():
    """Record live mic input to WAV file in background."""
    global recording
    filename = f"recordings/mic_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav"
    os.makedirs("recordings", exist_ok=True)

    def callback(indata, frames, time, status):
        if recording:
            record_queue.put(indata.copy())

    try:
        with sf.SoundFile(filename, mode='w', samplerate=16000, channels=1) as file:
            with sd.InputStream(samplerate=16000, channels=1, callback=callback):
                while recording:
                    if not record_queue.empty():
                        file.write(record_queue.get())
        return filename
    except Exception as e:
        print(f"Recording Error: {e}")
        return None

def start_recording():
    """Start a background thread for audio recording."""
    global recording
    recording = True
    threading.Thread(target=record_audio_wrapper).start()

def stop_recording():
    """Stop ongoing recording."""
    global recording
    recording = False

def record_audio_wrapper():
    """Run the audio recording, prediction, and plotting pipeline."""
    global recorded_file_path
    filepath = record_audio()
    recorded_file_path = filepath
    plot_all(filepath)
    run_prediction()

def _evaluate_voting_model():
    from joblib import load
    from emotion_pipeline.processing.utils.utils_train import load_and_preprocess
    from sklearn.metrics import classification_report, accuracy_score
    import webbrowser

    try:
        model_path = "models/voting_ensemble_balanced.joblib"
        if not os.path.isfile(model_path):
            messagebox.showerror("Model Not Found", f"{model_path} not found.")
            return

        # Load model and data
        model = load(model_path)
        _, X_test, _, y_test, _, encoder = load_and_preprocess()

        # Predict
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=encoder.classes_, digits=2)

        # ── Evaluation Results Window ─────────────────────
        # Open a separate evaluation window showing accuracy and classification metrics
        eval_window = tk.Toplevel(app)
        eval_window.title("Voting Ensemble Evaluation")
        eval_window.geometry("750x500")
        
        ttk.Label(
            eval_window,
            text=f"Voting Ensemble Evaluation\nAccuracy: {acc:.4f}",
            font=("Arial", 16, "bold")
        ).pack(pady=10)
        
        report_box = tk.Text(eval_window, height=18, width=90)
        report_box.pack(padx=10, pady=10)
        
        report_box.insert(
            tk.END,
            f"Classification Report:\n\n{report}"
        )
        
        report_box.config(state="disabled")
        
        ttk.Button(
            eval_window,
            text="Close",
            command=eval_window.destroy
        ).pack(pady=10)

        # Optionally open confusion matrix image
        cm_path = os.path.join("plots", "confusion", "confusion_matrix_voting.png")
        roc_path = os.path.join("plots", "roc", "roc_curves.png")
        if os.path.exists(cm_path):
            os.startfile(cm_path)
        if os.path.exists(roc_path):
            os.startfile(roc_path)

    except Exception as e:
        messagebox.showerror("Evaluation Error", str(e))

# ======== Trends & Leaderboard ========

def load_emotion_history(history):
    df = pd.DataFrame(history, columns=["timestamp", "emotion", "file"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df

def plot_weekly_trends(df):
    weekly_counts = df.groupby(["date", "emotion"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    weekly_counts.plot(kind="line", ax=ax, marker="o", title="Weekly Emotion Trend")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def show_leaderboard_chart(df):
    total_counts = Counter(df["emotion"])
    leaderboard = pd.DataFrame(total_counts.items(), columns=["Emotion", "Count"]).sort_values("Count", ascending=False)
    fig, ax = plt.subplots()
    leaderboard.plot(kind="bar", x="Emotion", y="Count", legend=False, ax=ax, title="🏆 Emotion Leaderboard")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def open_trends():
    if not emotion_history:
        messagebox.showinfo("Info", "No emotion history to analyze.")
        return
    df = load_emotion_history(emotion_history)
    plot_weekly_trends(df)

def open_leaderboard():
    if not emotion_history:
        messagebox.showinfo("Info", "No emotion history to rank.")
        return
    df = load_emotion_history(emotion_history)
    show_leaderboard_chart(df)

def play_current_audio():
    path = uploaded_file_path or recorded_file_path
    if not path:
        messagebox.showerror("Error", "No audio file to play.")
        return
    play_audio(path)

def clear_all():
    global uploaded_file_path, recorded_file_path, current_emotion, emotion_history

    uploaded_file_path = None
    recorded_file_path = None
    current_emotion = None
    emotion_history = []

    label_prediction.config(text="Emotion: N/A")
    result_label.config(text="")
    #reply_label.config(text="")
    text_response.delete("1.0", tk.END)
    box_history.delete(0, tk.END)

    for frame in [frame_wave, frame_spec, frame_pitch]:
        for widget in frame.winfo_children():
            widget.destroy()

# ======== GUI Widgets ========
ttk.Button(frame_top, text="Upload", command=upload_audio).pack(side='left', padx=5)
ttk.Button(frame_top, text="Play Audio", command=play_current_audio).pack(side='left', padx=5)
ttk.Button(frame_top, text="Predict Emotion", command=run_prediction).pack(side='left', padx=5)
ttk.Button(frame_top, text="ChatGPT Response", command=chat_response).pack(side='left', padx=5)
ttk.Button(frame_top, text="Start Mic", command=start_recording).pack(side='left', padx=5)
ttk.Button(frame_top, text="Stop Mic", command=stop_recording).pack(side='left', padx=5)
ttk.Button(frame_top, text="Sandbox", command=open_sandbox).pack(side='left', padx=5)
ttk.Button(frame_top, text="Weekly Trends", command=open_trends).pack(side='left', padx=5)
ttk.Button(frame_top, text="Leaderboard", command=open_leaderboard).pack(side='left', padx=5)
ttk.Button(frame_top, text="Evaluate Voting Model", command=_evaluate_voting_model).pack(side='left', padx=5)
ttk.Button(frame_top, text="Clear", command=clear_all).pack(side='left', padx=5)
label_prediction = ttk.Label(app, text="Emotion: N/A", font=("Arial", 16))
label_prediction.pack(pady=10)

result_label = ttk.Label(app, text="", font=("Arial", 14))
result_label.pack(pady=5)

#reply_label = ttk.Label(app, text="", font=("Arial", 12), wraplength=1000, justify="left")
#reply_label.pack(pady=5)

text_response = tk.Text(app, height=6, width=100)
text_response.pack(pady=10)

ttk.Label(app, text="Emotion History").pack()
box_history = tk.Listbox(app, height=8, width=100)
box_history.pack()

# ======== Main Loop ========
app.mainloop()