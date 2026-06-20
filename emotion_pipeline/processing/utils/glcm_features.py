"""
Media Design School - NLP303: Assignment 3  
Emotionally-Sensitive Agent Project  
Author: Rachel Heke  Id: MDS2000509  
Date: 23/06/2025  
Description: Extract GLCM (Gray-Level Co-occurrence Matrix) texture features from a spectrogram image.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image_path):
    """
    Extract 6 GLCM texture features from a grayscale image.

    Args:
        image_path (str or Path): Path to the input grayscale image (e.g., spectrogram).

    Returns:
        list[float]: List of texture features including:
            - Contrast
            - Dissimilarity
            - Homogeneity
            - ASM (Angular Second Moment)
            - Energy
            - Correlation
    """
    # Load image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Compute GLCM (offset=1 pixel, horizontal direction)
    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    # Extract selected texture features
    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]

    return features