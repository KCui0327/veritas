"""
Description: This script creates the Veritas dataset by running all processor scripts in the specified directory,
combining their outputs, and saving the final dataset to a CSV file

Author: Kenny Cui
Date: July 6, 2025
"""

import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from src.data_models.dataset import VeritasDataset
from src.data_models.dataset_emb import VeritasDatasetEmb
from src.util.logger import logger
from torch.nn.utils.rnn import pad_sequence

_DATASET_NAME = "data/veritas_dataset.csv"


def collate_fn(batch):
    """
    For each batch, pad the statements to the longest length in the batch.
    """
    statements = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    longest_statement_len = max(len(statement) for statement in statements)

    padded = []
    for statement in statements:
        if len(statement) < longest_statement_len:
            padding = longest_statement_len - len(statement)
            # 0 is <PAD> in our word2idx mapping
            zeros = torch.zeros(padding, dtype=torch.long)
            padded_statement = torch.cat((statement, zeros), dim=0)
        else:
            padded_statement = statement
        padded.append(padded_statement)

    ret_statements = torch.stack(padded)
    ret_labels = torch.stack(labels)

    return ret_statements, ret_labels


def collate_fn_embed(batch):
    embeddings, labels = zip(*batch)

    padded_embeddings = pad_sequence(
        embeddings,
        batch_first=True,
        padding_value=0.0,
    )

    labels = torch.stack(labels)

    return padded_embeddings, labels


def get_dataloaders(
    dataset_pth,
    train_size=0.8,
    batch_size=32,
    max_records=10000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for the training and validation sets.
    """
    if not dataset_pth:
        dataset_pth = _DATASET_NAME

    logger.info(f"Building dataset")

    if dataset_pth.endswith(".csv"):
        dataset = VeritasDataset(dataset_pth)
    elif dataset_pth.endswith(".npz"):
        dataset = VeritasDatasetEmb(dataset_pth)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_pth}")

    collate_func = collate_fn if not dataset_pth.endswith(".npz") else collate_fn_embed

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
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func
    )

    return train_loader, val_loader
