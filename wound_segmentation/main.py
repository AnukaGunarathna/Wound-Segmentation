"""
Main entry point for performing wound segmentation using a trained deep learning model.

This script provides a command-line interface (CLI) for running segmentation predictions
on either a single image or a directory of images. It performs preprocessing, inference,
and post-processing (mask overlay and saving results).

The script automatically downloads pretrained model weights from Google Drive if they are
not found locally.

Functions
---------
run_single_image(model, image_path, output_dir, threshold)
    Runs segmentation on a single image and saves the result.

run_batch(model, input_dir, output_dir, threshold)
    Runs segmentation on all valid images in a folder and saves the results.

main()
    Parses command-line arguments, loads the model, and executes the appropriate pipeline.

Examples
--------
Run inference on a single image:
>>> python main.py --image sample.jpg

Run inference on a folder of images:
>>> python main.py --input_dir data/images/

Notes
-----
- The model architecture is defined in 'model.py'.
- Preprocessing and mask saving utilities are used from 'preprocessing.py' and 'utils.py'.
- Weights are auto-downloaded from a public Google Drive link configured in 'constants.py'.

"""

import argparse
import os
import logging
import tensorflow as tf

from model import load_segmentation_model
from predict import predict_mask
from checkpoints import download_weights
from utils import save_result
from constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODEL_PATH,
    GDRIVE_FILE_ID,
    DEFAULT_MODEL_FILENAME,
)

# === Configure logging ===
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_single_image(
    model: tf.keras.Model, image_path: str, output_dir: str, threshold: float
) -> None:
    """
    Run segmentation on a single input image.

    This function runs the segmentation pipeline on a single image given as an input by the user.
    It calls the predict_mask function to predict the binary mask and saves the result to the
    given (or default) output directory.

    Parameters
    ----------
    model : tf.keras.Model
        The loaded segmentation model.
    image_path : str
        Path to the input RGB image file.
    output_dir : str
        Directory where the predicted result will be saved.
    threshold : float
        Threshold for binarizing the model's predicted probability map.

    Raises
    ------
    FileNotFoundError
        If the input image file does not exist.

    Examples
    --------
    >>> model = load_segmentation_model("model_weights/seg_model.weights.h5")
    >>> run_single_image(model, "images/sample.jpg", "outputs/", threshold=0.5)
    """
    # Ensure the image file exists before proceeding.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Run inference and generate a predicted mask.
    logging.info("Running prediction on image: %s", image_path)
    image, mask = predict_mask(model, image_path, threshold)

    # Create filename and save the prediction results.
    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_result(image, mask, basename, output_dir)

    # Log where the result was saved for traceability
    logging.info("Result saved to %s/%s_result.png", output_dir, basename)


def run_batch(
    model: tf.keras.Model, input_dir: str, output_dir: str, threshold: float
) -> None:
    """
    Run segmentation inference on all images in a given folder.

    This function runs the segmentation pipeline on a batch of images given as an input directory
    by the user. It calls the predict_mask function to predict the binary mask and saves the
    result to the given (or default) output directory.


    Parameters
    ----------
    model : tf.keras.Model
        The loaded segmentation model.
    input_dir : str
        Folder containing images to segment.
    output_dir : str
        Directory where the predicted results will be saved.
    threshold : float
        Threshold for binarizing the model's predicted probability map.

    Raises
    ------
    FileNotFoundError
        If the input directory is missing or contains no valid images.

    Examples
    --------
    >>> run_batch(model, "dataset/images", "outputs", threshold=0.6)
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Filter valid image files by extension
    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

    # Check if there are valid images to process
    if not image_files:
        raise FileNotFoundError(f"No valid images found in '{input_dir}'")

    logging.info("Found %d images in '%s'", len(image_files), input_dir)

    # Run prediction for each image in the directory
    for img_file in image_files:
        image_path = os.path.join(input_dir, img_file)
        try:
            run_single_image(model, image_path, output_dir, threshold)
        except ValueError as e:
            logging.warning("Skipping %s due to error: %s", img_file, e)


def main() -> None:
    """
    Parse command-line arguments and execute segmentation prediction.

    This function Supports single image mode (--image) and batch mode (--input_dir).
    It also automatically downloads model weights if not available locally.

    Raises
    ------
    SystemExit
        If CLI arguments are invalid or missing.

    Examples
    --------
    >>> python main.py --image image.jpg
    >>> python main.py --input_dir images/ --threshold 0.6
    """
    # Set up argument parser for CLI-based interaction
    parser = argparse.ArgumentParser(description="Segmentation Model Inference")

    # Optional: path to model weights (downloaded if missing)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model file",
    )

    # Allow either --image or --input_dir, not both
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single input image")
    group.add_argument("--input_dir", type=str, help="Directory of input images")

    # Optional: where to save output masks
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )

    # Optional: threshold to convert probabilities to binary mask
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to convert predicted probabilities to binary mask (default=0.5)",
    )

    # Parse the command-line inputs
    args = parser.parse_args()

    # If model file is missing, download it from Drive
    if not os.path.exists(args.model):
        logging.warning("Model not found at %s. Attempting to download...", args.model)
        download_weights(gdrive_id=GDRIVE_FILE_ID, model_name=DEFAULT_MODEL_FILENAME)

    # Load the trained model
    model = load_segmentation_model(args.model)

    # Choose between single or batch mode
    if args.image:
        run_single_image(model, args.image, args.output, args.threshold)
    elif args.input_dir:
        run_batch(model, args.input_dir, args.output, args.threshold)


if __name__ == "__main__":
    main()
