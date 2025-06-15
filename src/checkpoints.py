"""
Utilities for downloading and managing model checkpoints.

This module provides a function to automatically download pretrained model weights
from Google Drive if they are not found locally. The downloaded archive is extracted
into the default model directory, and the ZIP file is deleted afterward.

Logging is used to provide feedback during the download and extraction processes.
This module is typically used in the entry point script (main.py) to ensure model
weights are available before inference.

Functions
---------
download_weights(gdrive_id, model_name)
    Downloads zipped weights from Google Drive, extracts them to a target directory,
    and logs the progress. Raises RuntimeError if any step fails.

Examples
--------
>>> download_weights()
>>> model = load_segmentation_model("model_weights/segmentation_model.weights.h5")

Notes
-----
- The Google Drive ID and default model name are configured in 'constants.py'.
"""

import os
import logging
from zipfile import ZipFile
import gdown

from constants import GDRIVE_FILE_ID, DEFAULT_MODEL_DIR, DEFAULT_MODEL_FILENAME

# Module-level logger
logger = logging.getLogger(__name__)


def download_weights(
    gdrive_id: str = GDRIVE_FILE_ID, model_name: str = DEFAULT_MODEL_FILENAME
) -> None:
    """
    Download and save model weights from Google Drive.

    Downloads a zip file containing model weights using the specified Google Drive
    file ID, extracts the contents to the default model directory, and deletes the zip.

    Parameters
    ----------
    gdrive_id : str
        Google Drive file ID of the zipped weights.
    model_name : str
        Filename of the model weights (.weights.h5).

    Raises
    ------
    RuntimeError
        If download or extraction fails.

    Examples
    --------
    >>> download_weights()
    >>> model = load_segmentation_model("model_weights/segmentation_model.weights.h5")
    """
    zip_filename = f"{model_name}.zip"
    weights_filename = model_name
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), DEFAULT_MODEL_DIR
    )

    os.makedirs(model_dir, exist_ok=True)

    # Download the weights from Google Drive
    logger.info("Downloading %s weights... (takes about 4 minutes)", model_name)
    try:
        gdown.download(id=gdrive_id, output=zip_filename, quiet=True)
        logger.info("Download complete.")
    except Exception as e:
        logger.exception("Failed to download model weights.")
        raise RuntimeError(f"Failed to download model: {e}") from e

    # Extract the zip file
    logger.info("Extracting to '%s/'...", model_dir)
    try:
        with ZipFile(zip_filename, "r") as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(zip_filename)
        logger.info("Extraction complete.")
    except Exception as e:
        logger.exception("Failed to extract model weights.")
        raise RuntimeError(f"Failed to extract weights: {e}") from e

    logger.info("Model weights saved to %s", os.path.join(model_dir, weights_filename))
