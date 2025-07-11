"""
Description: This script creates the Veritas dataset by running all processor scripts in the specified directory,
combining their outputs, and saving the final dataset to a CSV file

Author: Kenny Cui
Date: July 6, 2025
"""

import os
from typing import Tuple

from torch.utils.data import DataLoader, Subset, random_split

from src.dataset.dataset import VeritasDataset

_DATASET_NAME = os.path.join(os.path.dirname(__file__), "veritas_dataset.csv")


def get_dataloaders(
    train_size=0.8,
    batch_size=32,
    max_records=10000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for the training and validation sets.
    """
    dataset = VeritasDataset(_DATASET_NAME)
    max_records = min(max_records, len(dataset))
    dataset = Subset(dataset, range(max_records))

    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Dataset info:")
    print(f"  Total dataset size: {len(dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader
