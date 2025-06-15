"""
Model architecture and loader for binary wound segmentation using U-Net with EfficientNetB3 encoder.

This module defines the core deep learning architecture for semantic segmentation. It includes:
- A function to construct a U-Net-style model using EfficientNetB3 as the encoder.
- A loader function that optionally loads pretrained weights into the model.

Functions
---------
- segmentation_model : Builds the segmentation model with skip connections and transposed
convolutions.
- load_segmentation_model : Loads the model with optional pretrained weights for inference
or fine-tuning.

Typical use
-----------
Used both for training and inference stages. The architecture is designed for detecting
wound areas in RGB images, and the loader supports loading '.weights.h5' weights
files for quick deployment.

"""

import tensorflow as tf
from tensorflow.keras import layers, models
from constants import IMG_SIZE


def segmentation_model(input_shape: tuple = (IMG_SIZE, IMG_SIZE, 3)) -> tf.keras.Model:
    """
    Build a U-Net style binary segmentation model with EfficientNetB3 as the encoder.

    This function builds a deep convolutional neural network model for semantic segmentation of
    images. The encoder is constructed using EfficientNetB3 model (without the top classification
    layer and without ImageNet pretrained weights) while the decoder block uses transpose
    convolutions and skip connections to upsample the feature maps.

    Parameters
    ----------
    input_shape : tuple
        Dimensions of the input RGB image, default (512, 512, 3)

    Returns
    -------
    tf.keras.Model
        A Keras model that maps input RGB images to binary masks with values in [0, 1]
        using sigmoid activation.

    Raises
    ------
    ValueError
        If the input shape has fewer than 3 dimensions

    Examples
    --------
    >>> model = segmentation_model(input_shape=(512, 512, 3))
    >>> model.summary()
    """
    if len(input_shape) < 3:
        raise ValueError(
            f"Expected input_shape with 3 dimensions, but got {input_shape}"
        )

    inputs = tf.keras.Input(shape=input_shape)

    # === Encoder ===
    # Use EfficientNetB3 as encoder without top classification layers and without pretrained weights
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False, weights=None, input_tensor=inputs
    )
    # Extract intermediate layers for skip connections
    skips = [
        base_model.get_layer(name).output
        for name in [
            "block2a_expand_activation",
            "block3a_expand_activation",
            "block4a_expand_activation",
            "block6a_expand_activation",
        ]
    ]

    x = base_model.output  # Last encoder output (low resolution)

    # === Decoder ===
    decoder_filters = [256, 128, 64, 32]

    for i, skip in enumerate(reversed(skips)):
        x = layers.Conv2DTranspose(decoder_filters[i], 3, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skip])  # Add skip connection from encoder
        x = layers.Conv2D(decoder_filters[i], 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

    # Final upsampling to get original resolution
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same")(x)
    # Final 1×1 convolution to output a single-channel binary mask
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output_mask")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def load_segmentation_model(weights_path: str = None) -> tf.keras.Model:
    """
    This function loads the binary segmentation model and apply the pre-trained weights.

    Parameters
    ----------
    weights_path : str
        Path to the pretrained model weights file. If None, the model is returned without pretrained
        weights.

    Returns
    -------
    tf.keras.Model
        The segmentation model with weights loaded if path is provided.

    Raises
    ------
    ValueError
        If the weights do not match the architecture or the file is corrupted.

    Examples
    --------
    >>> model = load_segmentation_model("models/segmentation_model_finetuned.keras")
    >>> pred_mask = model.predict(np.expand_dims(image, axis=0))
    """
    model = segmentation_model()

    if weights_path:
        try:
            model.load_weights(weights_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load weights from '{weights_path}': {e}"
            ) from e

    return model
