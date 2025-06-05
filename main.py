import argparse
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import load_segmentation_model
from predict import predict_mask

from constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MODEL_PATH,
    GDRIVE_FILE_ID,
    DEFAULT_MODEL_FILENAME,
)


def save_result(image, mask, basename, output_dir=DEFAULT_OUTPUT_DIR):
    """Save the input image and predicted mask side-by-side."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    ax[0].axis("off")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")
    output_path = os.path.join(output_dir, f"{basename}_result.png")
    plt.savefig(output_path)
    plt.close()


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

    model = load_segmentation_model(args.model)
    image, mask = predict_mask(model, args.image)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    save_result(image, mask, basename, args.output)
    print(f"Result saved to {args.output}/{basename}_result.png")


if __name__ == "__main__":
    main()
