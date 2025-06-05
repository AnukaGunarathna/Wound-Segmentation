import os
from zipfile import ZipFile
import gdown

from constants import GDRIVE_FILE_ID, DEFAULT_MODEL_DIR, DEFAULT_MODEL_FILENAME


def download_weights(
    gdrive_id: str = GDRIVE_FILE_ID, model_name: str = DEFAULT_MODEL_FILENAME
):
    zip_filename = f"{model_name}.zip"
    weights_filename = model_name
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), DEFAULT_MODEL_DIR
    )

    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading {model_name} weights...", end=" ", flush=True)
    try:
        gdown.download(id=gdrive_id, output=zip_filename, quiet=True)
        print("Done")
    except Exception as e:
        print("Failed")
        raise RuntimeError(f"Failed to download model: {e}") from e

    print(f"Extracting to '{model_dir}/'...", end=" ", flush=True)
    try:
        with ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(zip_filename)
        print("Done")
    except Exception as e:
        print("Failed")
        raise RuntimeError(f"Failed to extract weights: {e}") from e

    print(f"Model weights saved to {os.path.join(model_dir, weights_filename)}")
