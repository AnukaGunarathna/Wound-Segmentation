import numpy as np
import pytest
import tempfile
import cv2
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wound_segmentation"))
)

from predict import predict_mask
from tensorflow.keras import Model


class DummyModel(Model):
    """
    Dummy Keras model for testing. Always returns a constant mask of 0.6 values.
    """

    def predict(self, x):
        batch_size, height, width, _ = x.shape
        return np.ones((batch_size, height, width, 1)) * 0.6


@pytest.fixture
def temp_rgb_image():
    """
    Creates a temporary 512x512 RGB image and returns the file path.
    """
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, image)
    yield tmp.name
    os.remove(tmp.name)


def test_predict_mask_returns_image_and_binary_mask(temp_rgb_image):
    """
    GIVEN: a valid RGB image and dummy model
    WHEN: predict_mask is called
    THEN: it should return the preprocessed image and binary mask
    """
    model = DummyModel()
    image, mask = predict_mask(model, temp_rgb_image, threshold=0.5)
    assert isinstance(image, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert mask.ndim == 2
    assert set(np.unique(mask)).issubset({0, 1})


def test_predict_mask_raises_error_on_invalid_path():
    """
    GIVEN: an invalid image path
    WHEN: predict_mask is called
    THEN: it should raise FileNotFoundError
    """
    model = DummyModel()
    with pytest.raises(FileNotFoundError):
        predict_mask(model, "no_image.jpg")


def test_predict_mask_threshold_zero_all_ones(temp_rgb_image):
    """
    GIVEN: a model that returns 0.6 values and threshold=0.0
    WHEN: predict_mask is called
    THEN: the binary mask should contain only 1s
    """
    model = DummyModel()
    _, mask = predict_mask(model, temp_rgb_image, threshold=0.0)
    assert np.all(mask == 1)


def test_predict_mask_threshold_one_all_zeros(temp_rgb_image):
    """
    GIVEN: a model that returns 0.6 values and threshold=1.0
    WHEN: predict_mask is called
    THEN: the binary mask should contain only 0s
    """
    model = DummyModel()
    _, mask = predict_mask(model, temp_rgb_image, threshold=1.0)
    assert np.all(mask == 0)
