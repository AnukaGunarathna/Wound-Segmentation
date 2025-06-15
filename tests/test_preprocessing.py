import numpy as np
import pytest
import sys, os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wound_segmentation"))
)

from preprocessing import (
    resize_with_aspect_ratio,
    center_crop,
    scale_image_to_0_255,
    normalize_image,
)
from constants import IMG_SIZE


def test_resize_image_larger_and_shorter_sides():
    """
    GIVEN: a rectangular RGB image
    WHEN: passed to resize_with_aspect_ratio
    THEN: the shorter side should become IMG_SIZE, preserving aspect ratio
    """
    image = np.ones((300, 600, 3), dtype=np.uint8)
    resized = resize_with_aspect_ratio(image, IMG_SIZE)
    assert min(resized.shape[:2]) == IMG_SIZE


def test_resize_raises_on_non_array_input():
    """
    GIVEN: a non-numpy input
    WHEN: passed to resize_with_aspect_ratio
    THEN: it should raise ValueError
    """
    with pytest.raises(ValueError):
        resize_with_aspect_ratio("not_an_image")


def test_center_crop_returns_correct_shape():
    """
    GIVEN: a large image
    WHEN: center_crop is called
    THEN: it should return (IMG_SIZE, IMG_SIZE, 3)
    """
    image = np.ones((700, 800, 3), dtype=np.uint8)
    cropped = center_crop(image, IMG_SIZE)
    assert cropped.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_center_crop_on_image_smaller_than_crop_size():
    """
    GIVEN: an image that is smaller than the requested crop size
    WHEN: center_crop is called
    THEN: it should return the entire original image without error
    """
    # Image (400x400) is smaller than the crop size (512)
    image = np.ones((400, 400, 3), dtype=np.uint8)
    cropped = center_crop(image, IMG_SIZE)

    assert cropped.shape == image.shape


def test_scale_image_output_values_within_range():
    """
    GIVEN: an image with arbitrary range
    WHEN: scale_image_to_0_255 is called
    THEN: the output values should lie within [0, 255]
    """
    image = np.random.rand(256, 256) * 100
    scaled = scale_image_to_0_255(image)
    assert scaled.min() >= 0 and scaled.max() <= 255


def test_scale_image_to_0_255_raises_error_on_non_array():
    """
    GIVEN: a non-numpy input
    WHEN: passed to scale_image_to_0_255
    THEN: it should raise ValueError
    """
    with pytest.raises(ValueError):
        scale_image_to_0_255("not_a_numpy_array")


def test_normalize_image_channelwise_mean_zero():
    """
    GIVEN: an RGB image with noise
    WHEN: normalize_image is called
    THEN: the output should have zero mean per channel
    """
    image = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    norm = normalize_image(image)
    mean = np.mean(norm, axis=(0, 1))
    assert np.allclose(mean, 0.0, atol=1e-1)


def test_normalize_image_raises_error_on_non_rgb_input():
    """
    GIVEN: a grayscale image
    WHEN: normalize_image is called
    THEN: it should raise ValueError
    """
    gray = np.ones((512, 512), dtype=np.uint8)
    with pytest.raises(ValueError):
        normalize_image(gray)
