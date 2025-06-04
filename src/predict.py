import cv2
import numpy as np
from preprocessing import resize_with_aspect_ratio, center_crop, normalize_image


def predict_mask(model, image_path):
    """
    Predict a binary mask from an input image using the given segmentation model.

    Args:
        model (tf.keras.Model): Loaded segmentation model
        image_path (str): Path to the input image

    Returns:
        np.ndarray: Tuple of (original image, predicted mask)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_with_aspect_ratio(image)
    image = center_crop(image)
    image = normalize_image(image)
    input_tensor = image[np.newaxis, ...].astype(np.float32)

    predicted_mask = model.predict(input_tensor)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    return image, predicted_mask[..., 0]
