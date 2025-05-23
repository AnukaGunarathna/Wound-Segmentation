import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for binary segmentation tasks.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """
    Dice loss, defined as 1 - Dice coefficient.
    """
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss_weighted(y_true, y_pred, alpha=0.9):
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
    dsc = dice_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * dsc
