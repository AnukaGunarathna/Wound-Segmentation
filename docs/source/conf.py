# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Wound Segmentation"
copyright = "2025, Anuka Gunarathna"
author = "Anuka Gunarathna"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # supports NumPy-style docstrings
    "sphinx_rtd_theme",
    "myst_parser",  # only if using markdown in docs
]
autodoc_mock_imports = [
    "tensorflow",
    "numpy",
    "cv2",  # OpenCV
    "gdown",
    "matplotlib",
]
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = []

language = "y"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
