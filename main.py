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


def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Inference")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model file",
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )
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
    image, mask = predict_mask(model, args.image)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    save_result(image, mask, basename, args.output)
    print(f"Result saved to {args.output}/{basename}_result.png")


if __name__ == "__main__":
    main()
