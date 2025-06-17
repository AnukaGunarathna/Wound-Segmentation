import pytest
import tensorflow as tf

from wound_segmentation.model import segmentation_model, load_segmentation_model


def test_segmentation_model_output_shape():
    """
    GIVEN: a valid input shape
    WHEN: segmentation_model is called
    THEN: the output model should produce a mask of expected shape
    """
    model = segmentation_model(input_shape=(512, 512, 3))
    dummy_input = tf.random.normal((1, 512, 512, 3))
    output = model(dummy_input)
    assert output.shape == (1, 512, 512, 1)


def test_segmentation_model_raises_error_on_invalid_shape():
    """
    GIVEN: an input shape with fewer than 3 dimensions
    WHEN: segmentation_model is called
    THEN: it should raise ValueError
    """
    with pytest.raises(ValueError):
        segmentation_model(input_shape=(512, 512))


def test_load_segmentation_model_without_weights():
    """
    GIVEN: no weight path
    WHEN: load_segmentation_model is called
    THEN: it should return a valid model
    """
    model = load_segmentation_model()
    assert isinstance(model, tf.keras.Model)


def test_load_segmentation_model_invalid_weights(tmp_path):
    """
    GIVEN: an invalid weights file path
    WHEN: load_segmentation_model is called
    THEN: it should raise a ValueError
    """
    dummy_path = tmp_path / "no_file.h5"
    with pytest.raises(ValueError):
        load_segmentation_model(str(dummy_path))
