.. Wound Segmentation documentation master file, created by
   sphinx-quickstart on Mon Jun 16 14:02:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Wound Segmentation documentation !
==================================

**Wound Segmentation** is a deep learning-based tool developed to 
automatically detect and segment wound regions from clinical wound images.
This project leverages a U-Net–style architecture with an 
EfficientNetB3 encoder to provide accurate binary masks for wound 
boundaries. It is designed to be modular, reproducible and ready for deployment.

The system supports both single image and batch mode processing, and it 
is equipped with configurable parameters such as prediction thresholding 
and flexible input formats. All core functionalities—from preprocessing 
to model loading, inference, and result saving—are documented and covered 
by automated tests, ensuring reliability and transparency.

Key features include:

- **Pretrained model inference** using a robust U-Net backbone
- **Batch and single image prediction modes** for flexibility in evaluation
- **End-to-end preprocessing and postprocessing pipeline**, including aspect ratio preserving resizing, center cropping, and overlay visualization.
- **Command-line interface** for ease of integration into larger workflows
- **Test coverage for all critical components** with high code quality and exception handling
- **Fully auto-generated API documentation** using Sphinx for maintainability
- **Lightweight**—can run locally with minimal setup

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
