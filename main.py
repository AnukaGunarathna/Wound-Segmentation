import argparse
import os
import sys
import matplotlib.pyplot as plt

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


def run_single_image(model, image_path, output_dir):
    image, mask = predict_mask(model, image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_result(image, mask, basename, output_dir)
    print(f"Result saved to {output_dir}/{basename}_result.png")


def run_batch(model, input_dir, output_dir):
    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in '{input_dir}'")

    for img_file in image_files:
        image_path = os.path.join(input_dir, img_file)
        run_single_image(model, image_path, output_dir)


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
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("Downloading model weights...")
        download_weights(gdrive_id=GDRIVE_FILE_ID, model_name=DEFAULT_MODEL_FILENAME)

    model = load_segmentation_model(args.model)

    if args.image:
        run_single_image(model, args.image, args.output)
    elif args.input_dir:
        run_batch(model, args.input_dir, args.output)


if __name__ == "__main__":
    main()
