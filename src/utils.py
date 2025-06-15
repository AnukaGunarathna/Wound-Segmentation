"""
Utility functions for wound segmentation evaluation and result visualization.

This module contains helper functions used throughout the segmentation pipeline,
including:

- Metric computation (e.g., Dice coefficient and Dice-based loss)
- Combined loss function (BCE + Dice) for model training.
- Visualization and saving of segmentation results.

These utilities are designed to be modular and testable.

Functions
---------
- dice_coef : Compute Dice similarity coefficient between two binary masks.
- dice_loss : Calculate Dice loss from Dice coefficient.
- bce_dice_loss_weighted : Weighted combination of BCE and Dice losses.
- save_result : Generate and save a 3-panel plot (input, mask, overlay).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from preprocessing import scale_image_to_0_255


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Compute the Dice coefficient, a measure of similarity between two binary masks.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth binary mask.
    y_pred : tf.Tensor
        Predicted binary mask.
    smooth : float, optional
        Smoothing constant to avoid division by zero (default is 1e-6).

    Returns
    -------
    tf.Tensor
        Dice similarity coefficient.

    Raises
    ------
    TypeError
        If `y_true` or `y_pred` is not a tf.Tensor.

    Examples
    --------
    >>> dice_coef(y_true, y_pred)
    """
    if not isinstance(y_true, tf.Tensor) or not isinstance(y_pred, tf.Tensor):
        raise TypeError("Both y_true and y_pred must be TensorFlow tensors.")

    # Flatten the masks
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Compute intersection and Dice score
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_score = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return dice_score


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute the Dice loss, defined as 1 - Dice coefficient.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth binary mask.
    y_pred : tf.Tensor
        Predicted binary mask.

    Returns
    -------
    tf.Tensor
        Dice loss value in the range [0, 1].

    Examples
    --------
    >>> dice = dice_loss(y_true, y_pred)
    """
    loss_value = 1.0 - dice_coef(y_true, y_pred)
    return loss_value


def bce_dice_loss_weighted(
    y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.2
) -> tf.Tensor:
    """
    Compute a weighted combination of Binary Crossentropy and Dice loss.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth binary mask.
    y_pred : tf.Tensor
        Predicted binary mask..
    alpha : float, optional
        Weight for BCE loss; (1 - alpha) is used for Dice loss.

    Returns
    -------
    tf.Tensor
        Weighted loss value.

    Raises
    ------
    ValueError
        If alpha is not in the range [0, 1].

    Examples
    --------
    >>> bce_dice_loss_weighted(y_true, y_pred, alpha=0.3)
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    # Compute BCE and Dice loss separately
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    # Compute the weighted combination
    weighted_loss = alpha * bce + (1 - alpha) * dice
    return weighted_loss


def save_result(
    image: np.ndarray, mask: np.ndarray, basename: str, output_dir: str
) -> None:  # pragma: no cover
    """
    Save the input image and predicted mask side-by-side as a PNG.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image of shape (H, W, 3).
    mask : np.ndarray
        Binary predicted mask.
    basename : str
        Output image file name prefix.
    output_dir : str
        Directory path to save the result image.

    Raises
    ------
    OSError
        If the output directory cannot be created.

    Examples
    --------
    >>> save_result(image, mask, "sample", "outputs/")
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        raise OSError(f"Could not create output directory: {output_dir}") from e

    display_image = scale_image_to_0_255(image)

    # Define overlay mask features.
    overlay_mask_color = cv2.cvtColor(np.uint8(mask * 255), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(display_image, 0.7, overlay_mask_color, 0.3, 0)

    # Plot and save the comparison figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(display_image)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    ax[2].imshow(display_image)
    ax[2].imshow(overlay)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    output_path = os.path.join(output_dir, f"{basename}_result.png")
    plt.savefig(output_path)
    plt.close()
