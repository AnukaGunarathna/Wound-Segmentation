sourceimport argparse
import os
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import matplotlib.pyplot as plt
from model import load_segmentation_model
from predict import predict_mask


def save_result(image, mask, output_dir, basename):
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
        "--model", type=str, default="../model_weights/segmentation_model.weights.h5", help="Path to the model file"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--output", type=str, default="outputs", help="Directory to save results"
    )
    args = parser.parse_args()

    model = load_segmentation_model(args.model)
    image, mask = predict_mask(model, args.image)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    save_result(image, mask, args.output, basename)
    print(f"Result saved to {args.output}/{basename}_result.png")


if __name__ == "__main__":
    main()
