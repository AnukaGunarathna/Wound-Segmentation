"""
Model inference utilities for generating segmentation masks from input images.

This module provides functions for running a trained segmentation model on a
single input image. It includes functionality to load the image, apply the
same preprocessing used during training, run inference, and convert the output
into a binary mask.

Functions
---------
- predict_mask : This applies preprocessing, runs model prediction, and returns both
  the preprocessed image and the predicted mask.

Typical use
-----------
This is used during evaluation or inference time when applying the trained model to
new wound images for mask prediction. It handles file loading, normalization,
and shape adjustments to ensure compatibility with the model input.
"""

import cv2
import numpy as np
import tensorflow as tf

from .preprocessing import resize_with_aspect_ratio, center_crop, normalize_image


def predict_mask(
    model: tf.keras.Model, image_path: str, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict a binary mask from an input image using the given segmentation model.

    This function loads an image, applies the preprocessing steps used during training
    (resize, center-crop, normalize), runs the model prediction, and thresholds
    the result to produce a binary mask.

    Parameters
    ----------
    model : tf.keras.Model
        The loaded segmentation model.
    image_path : str
        Path to the input image file.
    threshold : float, optional
        Threshold applied to the model's output to binarize the mask. Default is 0.5.

    Returns
    -------
    image : np.ndarray
        Preprocessed RGB image.
    mask : np.ndarray
        Binary mask with values {0, 1}.

    Raises
    ------
    FileNotFoundError
        If the input image cannot be read from the provided path.

    Examples
    --------
    >>> model = load_segmentation_model("weights_path")
    >>> image, mask = predict_mask(model, "image.jpg", threshold=0.6)
    """
    # Load image from file
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image from: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply same preprocessing steps as in training
    image = resize_with_aspect_ratio(image)
    image = center_crop(image)
    image = normalize_image(image)

    # Add batch dimension
    input_tensor = image[np.newaxis, ...].astype(np.float32)

    # Run model prediction
    predicted_mask = model.predict(input_tensor)[0]

    # Binarize output using threshold
    predicted_mask = (predicted_mask > threshold).astype(np.uint8)

    return image, predicted_mask[..., 0]
