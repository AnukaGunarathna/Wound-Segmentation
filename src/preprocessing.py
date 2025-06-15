"""
Preprocessing functions for wound image resizing, cropping and normalization,

This module includes core image preprocessing utilities used during both training
and inference phases of the wound segmentation pipeline. All functions are
designed to ensure shape consistency, pixel scaling, and model compatibility.

Functions
---------
- resize_with_aspect_ratio : Resize images while preserving aspect ratio and padding if needed.
- center_crop : Crop a square patch from the center of an image.
- scale_image_to_0_255 : Rescale image values to [0, 255] range and convert to uint8.
- normalize_image : Normalize an RGB image to zero mean and unit variance.

These transformations ensure model robustness across varying input sizes and lighting conditions.
"""

import cv2
import numpy as np
from constants import IMG_SIZE


def resize_with_aspect_ratio(
    image: np.ndarray, target_size: int = IMG_SIZE
) -> np.ndarray:
    """
    Resize an image so the shorter side becomes `target_size`, preserving aspect ratio.
    Pads with zeros to avoid shape mismatches due to rounding. The function can handles
    both grayscale and RGB images.

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB or grayscale).
    target_size : int
        Desired size of the shorter side after resizing. Default is IMG_SIZE.

    Returns
    -------
    np.ndarray
        Resized (and padded) image.

    Raises
    ------
    ValueError
        If the input is not a NumPy array.

    Examples
    --------
    >>> resized = resize_with_aspect_ratio(img, IMG_SIZE)
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a numpy array.")

    # Get original height and width
    h, w = image.shape[:2]

    # Scale based on the shorter side
    scale = target_size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image while maintaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding needed to reach target size
    pad_h = max(0, target_size - resized.shape[0])
    pad_w = max(0, target_size - resized.shape[1])

    # Apply zero-padding if necessary
    if pad_h > 0 or pad_w > 0:  # pragma: no cover
        if resized.ndim == 3:
            # RGB image
            resized = np.pad(
                resized,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            # Grayscale mask
            resized = np.pad(
                resized,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )
    return resized


def center_crop(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    This function crops a square region of shape (size, size) from the center of the image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB).
    size : int
        Desired size of the square crop.

    Returns
    -------
    np.ndarray
        Center-cropped image.

    Examples
    --------
    >>> crop = center_crop(np.ones(img, IMG_SIZE)
    """
    h, w = image.shape[:2]

    # Compute starting points for cropping
    start_y = max(0, h // 2 - size // 2)
    start_x = max(0, w // 2 - size // 2)

    # Crop a centered region
    cropped_image = image[start_y : start_y + size, start_x : start_x + size]
    return cropped_image


def scale_image_to_0_255(image: np.ndarray) -> np.ndarray:
    """
    Rescale an image to [0, 255] range and convert to uint8.

    Parameters
    ----------
    image : np.ndarray
        Input image of arbitrary range.

    Returns
    -------
    np.ndarray
        Image scaled to uint8 [0, 255].

    Raises
    ------
    ValueError
        If input is not a NumPy array.

    Examples
    --------
    >>> scaled = scale_image_to_0_255(img)
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    # Normalize to [0, 1]
    image_min = np.min(image)
    image_max = np.max(image)
    scaled_to_0_1 = (image - image_min) / (image_max - image_min + 1e-7)

    # Scale to [0, 255] and cast to uint8
    scaled_to_0_255 = (scaled_to_0_1 * 255).astype(np.uint8)
    return scaled_to_0_255  # scale to [0, 255]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to have zero mean and unit standard deviation per channel.
    Normalization removes highlighted regions, shadows and make that object easier to detect.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image.

    Returns
    -------
    np.ndarray
        Image normalized per channel.

    Raises
    ------
    ValueError
        If input is not a 3D RGB image.

    Examples
    --------
    >>> normalized = normalize_image(image)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    # Rescale to [0, 255] for consistency
    image = scale_image_to_0_255(image)

    # Compute channel-wise mean and std
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)

    # Normalize each channel
    normalized_image = (image - mean) / (
        std + 1e-7
    )  # 1e-7 is added to avoid division by zero
    return normalized_image
