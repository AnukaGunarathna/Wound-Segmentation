"""
Global constants used across the segmentation project.

This module defines all key constants for consistent usage throughout the codebase,
such as image size, default filenames, output paths, and download identifiers.

Variables
---------
IMG_SIZE : int
    Input image size used for both model training and inference (square dimension).
DEFAULT_MODEL_FILENAME : str
    Default filename for the segmentation model weights.
DEFAULT_MODEL_DIR : str
    Directory where model weights are stored or downloaded to.
DEFAULT_MODEL_PATH : str
    Full path to the model weights file (directory + filename).
DEFAULT_OUTPUT_DIR : str
    Directory where prediction results and overlay images are saved.
GDRIVE_FILE_ID : str
    Google Drive ID used to download zipped model weights if missing locally.

Usage
-----
These constants are imported by different modules to ensure unified configuration
and path management.

Examples
--------
>>> from constants import IMG_SIZE, DEFAULT_MODEL_PATH
>>> print(DEFAULT_MODEL_PATH)
'model_weights/segmentation_model.weights.h5'
"""

# Input image size for model
IMG_SIZE = 512

# Default file and directory names
DEFAULT_MODEL_FILENAME = "segmentation_model.weights.h5"
DEFAULT_MODEL_DIR = "model_weights"
DEFAULT_MODEL_PATH = f"{DEFAULT_MODEL_DIR}/{DEFAULT_MODEL_FILENAME}"

DEFAULT_OUTPUT_DIR = "outputs"

# Google Drive ID of the zipped model weights
GDRIVE_FILE_ID = "1Rldcue5dVF2XTp1kx3G5hBvG3xApwSaS"
