"""
Description: This script creates the Veritas dataset by running all processor scripts in the specified directory,
combining their outputs, and saving the final dataset to a CSV file

Author: Kenny Cui
Date: July 6, 2025
"""

import os
import re
from collections import Counter
from typing import Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_DATASET_NAME = os.path.join(os.path.dirname(__file__), "veritas_dataset.csv")


def preprocess_text(text):
    """Clean and tokenize text."""
    if not isinstance(text, str):
        return []

    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = text.split()
    return tokens


def build_vocabulary(data_frame: pd.DataFrame, vocab_size: int) -> Tuple[dict, dict]:
    """Build vocabulary from all texts in the dataset."""
    word_counts = Counter()

    for _, row in data_frame.iterrows():
        tokens = preprocess_text(row["statement"])
        word_counts.update(tokens)

    most_common = word_counts.most_common(vocab_size - 2)  # -2 for <PAD> and <UNK>

    word2idx = {"<PAD>": 0, "<UNK>": 1}
    word2idx.update({word: idx + 2 for idx, (word, _) in enumerate(most_common)})

    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word


def text_to_tensor(text, word2idx, max_length) -> torch.Tensor:
    """Convert text to tensor of token indices."""
    tokens = preprocess_text(text)

    # Convert tokens to indices
    indices = []
    for token in tokens:
        if token in word2idx:
            indices.append(word2idx[token])
        else:
            indices.append(word2idx["<UNK>"])

    # Pad or truncate to max_length
    if len(indices) < max_length:
        indices.extend([word2idx["<PAD>"]] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]

    return torch.tensor(indices, dtype=torch.long)


def get_dataloaders(
    validation_size=0.2,
    batch_size=32,
    max_token_length=512,
    max_training_records=1000,
    max_validation_records=1000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get the dataloaders for the training and validation sets.
    """
    data_frame = pd.read_csv(_DATASET_NAME)
    word2idx, _ = build_vocabulary(data_frame, vocab_size=20000)

    # Get all data as tensors
    all_inputs, all_labels = [], []
    for i in range(len(data_frame)):
        input_tensor = text_to_tensor(
            data_frame["statement"][i], word2idx, max_token_length
        )
        all_inputs.append(input_tensor)
        all_labels.append(0 if data_frame["verdict"][i] == "True" else 1)

    # Stratified split of the dataset
    x_train, x_val, y_train, y_val = train_test_split(
        all_inputs,
        all_labels,
        test_size=validation_size,
        random_state=1,  # Seed
        stratify=all_labels,  # Enable Stratified split
    )

    # limit x_train and x_val to max_records
    x_train = x_train[:max_training_records]
    y_train = y_train[:max_training_records]

    x_val = x_val[:max_validation_records]
    y_val = y_val[:max_validation_records]

    train_dataset = TensorDataset(
        torch.stack(x_train), torch.tensor(y_train, dtype=torch.float)
    )
    val_dataset = TensorDataset(
        torch.stack(x_val), torch.tensor(y_val, dtype=torch.float)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
