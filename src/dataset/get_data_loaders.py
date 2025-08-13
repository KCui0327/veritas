"""
Description: This script creates the Veritas dataset by running all processor scripts in the specified directory,
combining their outputs, and saving the final dataset to a CSV file

Author: Kenny Cui
Date: July 6, 2025
"""

import random
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset, random_split

from src.data_models.dataset import VeritasDataset
from src.util.logger import logger

_DATASET_NAME = "data/veritas_dataset.csv"
_TEST_DATASET_NAME = "data/veritas_dataset_test.csv"


def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return pad_sequence(inputs, batch_first=True, padding_value=0.0), torch.stack(
        labels
    )


def get_dataloaders(
    train_size=0.8,
    batch_size=32,
    max_records=100_000,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get the dataloaders for the training and validation sets.
    """
    logger.info(f"Building dataset")
    dataset = VeritasDataset(_DATASET_NAME)

    # Shuffle the dataset before subsetting
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    dataset = Subset(dataset, indices)

    # Only take the first max_records records
    max_records = min(max_records, len(dataset))
    dataset = Subset(dataset, range(max_records))

    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Total dataset size: {len(dataset)}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=32) -> DataLoader:
    test_dataset = VeritasDataset(_TEST_DATASET_NAME)
    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
