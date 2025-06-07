import argparse
import os
import sys
import logging

# import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
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


def run_single_image(model, image_path, output_dir, threshold):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    logging.info("Running prediction on image: %s", image_path)
    image, mask = predict_mask(model, image_path, threshold)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_result(image, mask, basename, output_dir)
    logging.info("Result saved to %s/%s_result.png", output_dir, basename)


def run_batch(model, input_dir, output_dir, threshold):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

    if not image_files:
        raise FileNotFoundError(f"No valid images found in '{input_dir}'")

    logging.info("Found %d images in '%s'", len(image_files), input_dir)

    for img_file in image_files:
        image_path = os.path.join(input_dir, img_file)
        try:
            run_single_image(model, image_path, output_dir, threshold)
        except ValueError as e:
            logging.warning("Skipping %s due to error: %s", img_file, e)


def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Inference")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model file",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single input image")
    group.add_argument("--input_dir", type=str, help="Directory of input images")

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to convert predicted probabilities to binary mask (default=0.5)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        logging.warning("Model not found at %s. Attempting to download...", args.model)
        download_weights(gdrive_id=GDRIVE_FILE_ID, model_name=DEFAULT_MODEL_FILENAME)

    model = load_segmentation_model(args.model)

    if args.image:
        run_single_image(model, args.image, args.output, args.threshold)
    elif args.input_dir:
        run_batch(model, args.input_dir, args.output, args.threshold)


if __name__ == "__main__":
    main()
