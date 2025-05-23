from tensorflow.keras.models import load_model
from utils import dice_coef, bce_dice_loss_weighted


def load_segmentation_model(model_path):
    """
    Load a Keras model with custom loss and metric functions.

    Args:
        model_path (str): Path to the .keras or .h5 model file

    Returns:
        tf.keras.Model: Compiled model ready for prediction or evaluation
    """
    model = load_model(
        model_path,
        custom_objects={
            "bce_dice_loss_weighted": bce_dice_loss_weighted,
            "dice_coef": dice_coef,
        },
    )
    return model
