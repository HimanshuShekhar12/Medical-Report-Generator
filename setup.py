"""
setup.py
--------
Makes MedReportGen importable as a package.

After running: pip install -e .
You can do:   from src.data import XRayDataset
from anywhere in the project without path hacks.

Install with:
  pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name         = "medreportgen",
    version      = "0.1.0",
    description  = "Multimodal Medical Report Generator using DDPM + VAE + BioGPT",
    author       = "Himanshu Shekhar",
    packages     = find_packages(),
    python_requires = ">=3.9",
    install_requires = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0.0",
    ],
)