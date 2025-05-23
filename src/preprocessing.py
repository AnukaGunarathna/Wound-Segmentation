import cv2
import numpy as np


def resize_with_aspect_ratio(image, target_size=512):
    """
    Resize an image such that the shorter side becomes `target_size`, preserving aspect ratio.
    """
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # an error occured where one size resized to 511. Hence the padding steps
    # Pad to ensure both sides are at least 512
    pad_h = max(0, target_size - resized.shape[0])
    pad_w = max(0, target_size - resized.shape[1])

    if pad_h > 0 or pad_w > 0:
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


# Random Augmentations (Shift, Rotate, Flip)
# Increases diversity in training data and helps prevent overfitting.
"""
# ====== NOT NEEDED FOR PREDICTIONS ====== #
def augment_image_and_mask(image, mask):
    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    # Random vertical flip
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    # Random rotation
    angle = np.random.uniform(-30, 30)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    return image, mask

"""


def center_crop(image, size=512):
    """
    Center crop a square patch from the image of size `target_size`.
    Assumes image is larger than target_size.
    """
    h, w = image.shape[:2]
    start_y = max(0, h // 2 - size // 2)
    start_x = max(0, w // 2 - size // 2)
    return image[start_y : start_y + size, start_x : start_x + size]


# RGB Normalization (Image-Wise Mean & Std)
# Normalization removes highlighted regions, shadows and make that object easier to detect
def normalize_image(image):
    # Compute per-channel mean and std
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    return (image - mean) / (std + 1e-7)  # 1e-7 is added to avoid division by zero
