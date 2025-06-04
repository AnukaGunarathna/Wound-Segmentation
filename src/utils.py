import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for binary segmentation tasks.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_score = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return dice_score


def dice_loss(y_true, y_pred):
    """
    Dice loss, defined as 1 - Dice coefficient.
    """
    loss_value = 1.0 - dice_coef(y_true, y_pred)
    return loss_value


def bce_dice_loss_weighted(y_true, y_pred, alpha=0.2):
    """
    Weighted combination of Binary Crossentropy and Dice Loss.

    Args:
        y_true: ground truth mask
        y_pred: predicted mask
        alpha: weight for BCE loss; (1-alpha) used for Dice loss
    Returns:
        Combined weighted loss
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    weighted_loss = alpha * bce + (1 - alpha) * dice
    return weighted_loss
