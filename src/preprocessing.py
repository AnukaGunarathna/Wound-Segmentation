import cv2
import numpy as np
from constants import IMG_SIZE


def resize_with_aspect_ratio(image, target_size=IMG_SIZE):
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


def center_crop(image, size=IMG_SIZE):
    """
    Center crop a square patch from the image of size `target_size`.
    Assumes image is larger than target_size.
    """
    h, w = image.shape[:2]
    start_y = max(0, h // 2 - size // 2)
    start_x = max(0, w // 2 - size // 2)
    cropped_image = image[start_y : start_y + size, start_x : start_x + size]
    return cropped_image


def scale_image_to_0_255(image):
    image_min = np.min(image)
    image_max = np.max(image)
    scaled_to_0_1 = (image - image_min) / (
        image_max - image_min + 1e-7
    )  # scale to [0, 1]
    scaled_to_0_255 = (scaled_to_0_1 * 255).astype(np.uint8)
    return scaled_to_0_255  # scale to [0, 255]


# RGB Normalization (Image-Wise Mean & Std)
# Normalization removes highlighted regions, shadows and make that object easier to detect
def normalize_image(image):
    # Compute per-channel mean and std
    image = scale_image_to_0_255(image)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    normalized_image = (image - mean) / (
        std + 1e-7
    )  # 1e-7 is added to avoid division by zero
    return normalized_image
