"""
Media Design School - NLP303: Assignment 3  
Emotionally-Sensitive Agent Project  
Author: Rachel Heke  Id: MDS2000509  
Date: 23/06/2025  
Description: Extract HOG (Histogram of Oriented Gradients) features from a spectrogram image.
"""

import cv2
from skimage.feature import hog

def extract_hog_features(image_path):
    """
    Extract Histogram of Oriented Gradients (HOG) features from a grayscale image.

    Args:
        image_path (str or Path): Path to the input image (e.g., spectrogram).

    Returns:
        np.ndarray: Truncated HOG feature vector (first 50 components).
    """
    # Load image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Resize to standard size for HOG computation consistency
    img_resized = cv2.resize(img, (128, 128))

    # Compute HOG descriptor
    hog_features = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # Return the first 50 HOG features for dimensionality control
    return hog_features[:50]