"""
This code package contains modules for processing datasets and creating data loaders.
"""

from src.dataset.run import run as get_dataloaders
from src.dataset.run import split_dataset

__all__ = ["get_dataloaders", "split_dataset"]
