#!/usr/bin/env python3
"""
Media Design School - NLP303: Assignment 3
Emotionally-Sensitive Agent Project
Author: Rachel Heke  Id: MDS2000509
Date: 23/06/2025
Description: Utility functions to embed Matplotlib plots into a Tkinter GUI.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def embed_plot(fig, target_frame):
    """
    Embed a Matplotlib figure into a specified Tkinter frame.

    Args:
        fig (matplotlib.figure.Figure): The Matplotlib figure to display.
        target_frame (tkinter.Frame): The frame in which to embed the plot.
    """
    # Clear any existing widgets in the target frame
    for widget in target_frame.winfo_children():
        widget.destroy()
    
    # Create a Tkinter-compatible canvas with the figure
    canvas = FigureCanvasTkAgg(fig, master=target_frame)
    canvas.draw()  # Render the plot
    
    # Embed the canvas widget in the GUI frame
    canvas.get_tk_widget().pack(fill='both', expand=True)

def show_pca_plot(frame):
    """
    Example function to display a simple PCA line plot within a Tkinter frame.

    Args:
        frame (tkinter.Frame): The GUI frame where the plot will appear.
    """
    # Create a simple line plot for demonstration
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])  # Dummy data
    ax.set_title("Sample PCA Plot")
    
    # Embed the plot in the specified frame
    embed_plot(fig, frame)