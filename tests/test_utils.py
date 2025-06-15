import numpy as np
import tensorflow as tf
import pytest
import sys, os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wound_segmentation"))
)
from utils import dice_coef, dice_loss, bce_dice_loss_weighted


def test_dice_coef_perfect_match():
    """
    GIVEN: two identical binary masks
    WHEN: passed to dice_coef
    THEN: the Dice score should be approximately 1.0
    """
    y = tf.ones((128, 128, 1))
    score = dice_coef(y, y)
    assert tf.abs(score - 1.0) < 1e-6


def test_dice_coef_disjoint_masks():
    """
    GIVEN: a mask of ones and a mask of zeros
    WHEN: passed to dice_coef
    THEN: the Dice score should be approximately 0.0
    """
    y_true = tf.ones((128, 128, 1))
    y_pred = tf.zeros((128, 128, 1))
    score = dice_coef(y_true, y_pred)
    assert tf.abs(score) < 1e-6


def test_dice_coef_raises_type_error_on_non_tensor_input():
    """
    GIVEN: inputs that are not tf.Tensors
    WHEN: passed to dice_coef
    THEN: it should raise a TypeError
    """
    y_true = np.ones((128, 128, 1))
    y_pred = np.ones((128, 128, 1))
    with pytest.raises(TypeError):
        dice_coef(y_true, y_pred)


def test_dice_loss_perfect_match():
    """
    GIVEN: two identical binary masks
    WHEN: passed to dice_loss
    THEN: the Dice loss should be approximately 0.0
    """
    y = tf.ones((64, 64, 1))
    loss = dice_loss(y, y)
    assert tf.abs(loss) < 1e-6


def test_dice_loss_disjoint_masks():
    """
    GIVEN: a mask of ones and a mask of zeros
    WHEN: passed to dice_loss
    THEN: the Dice loss should be approximately 1.0
    """
    y_true = tf.ones((64, 64, 1))
    y_pred = tf.zeros((64, 64, 1))
    loss = dice_loss(y_true, y_pred)
    assert tf.abs(loss - 1.0) < 1e-6


def test_weighted_loss_with_alpha_zero():
    """
    GIVEN: two masks and alpha = 0
    WHEN: passed to bce_dice_loss_weighted
    THEN: the result should equal Dice loss
    """
    y_true = tf.ones((32, 32, 1))
    y_pred = tf.ones((32, 32, 1))
    expected = dice_loss(y_true, y_pred)
    loss = bce_dice_loss_weighted(y_true, y_pred, alpha=0.0)
    assert tf.abs(loss - expected) < 1e-6


def test_weighted_loss_with_alpha_one():
    """
    GIVEN: two masks and alpha = 1
    WHEN: passed to bce_dice_loss_weighted
    THEN: the result should equal BCE loss
    """
    y_true = tf.ones((32, 32, 1))
    y_pred = tf.ones((32, 32, 1))
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    loss = bce_dice_loss_weighted(y_true, y_pred, alpha=1.0)
    assert tf.abs(loss - bce) < 1e-6


def test_weighted_loss_raises_on_invalid_alpha():
    """
    GIVEN: an invalid alpha > 1
    WHEN: passed to bce_dice_loss_weighted
    THEN: it should raise ValueError
    """
    y_true = tf.ones((32, 32, 1))
    y_pred = tf.ones((32, 32, 1))
    with pytest.raises(ValueError):
        bce_dice_loss_weighted(y_true, y_pred, alpha=1.5)
