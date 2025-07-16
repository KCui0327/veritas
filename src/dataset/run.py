"""
Description: This script creates the Veritas dataset by running all processor scripts in the specified directory,
combining their outputs, and saving the final dataset to a CSV file

Author: Kenny Cui
Date: July 6, 2025
"""

import glob
import hashlib
import os
import runpy

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset.dataset import VeritasDataset

# Get absolute paths
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH = os.path.join(_CURRENT_DIR, "output")
_DATASET_NAME = os.path.join(_CURRENT_DIR, "veritas_dataset.csv")
_PROCESSOR_PATH = os.path.join(_CURRENT_DIR, "processor")
_DATA_PATH = os.path.join(_CURRENT_DIR, "data")


def process_data():
    for file in os.listdir(_PROCESSOR_PATH):
        if file.endswith(".py"):
            print(f"Running processor: {file}")
            runpy.run_path(os.path.join(_PROCESSOR_PATH, file))

    dataset_files = glob.glob(f"{_DATASET_PATH}/*.csv")
    datasets = []
    for file in dataset_files:
        print(f"Processing {file}")
        df = pd.read_csv(file, header=0)
        datasets.append(df)

    df = pd.concat(datasets, ignore_index=True)
    # random state is seed
    df = df.sample(frac=1, random_state=21).reset_index(
        drop=True
    )  # Shuffle the DataFrame

    # Remove duplicates based on the 'statement' column through unique hashing
    df["id"] = df["statement"].apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )
    df.drop_duplicates(subset=["id"], inplace=True, keep="first")

    df.to_csv(_DATASET_NAME, index=False, encoding="utf-8")


def split_dataset(validation_size=0.2):
    # Create a VeritasDataset instance
    dataset = VeritasDataset(_DATASET_NAME)

    # Get all data as tensors
    all_inputs, all_labels = [], []
    for i in range(len(dataset)):
        input_tensor, label_tensor = dataset[i]
        all_inputs.append(input_tensor)
        all_labels.append(label_tensor)

    # Stratified split of the dataset
    x_train, x_val, y_train, y_val = train_test_split(
        all_inputs,
        all_labels,
        test_size=validation_size,
        random_state=1,  # Seed
        stratify=all_labels,  # Enable Stratified split
    )

    return x_train, x_val, y_train, y_val


def create_loader(
    x, y, batch_size=32, shuffle=False, num_workers=0, word2idx=None, idx2word=None
):
    # Create a new VeritasDataset instance for the split data
    # Convert tensors back to text format for the dataset constructor
    statements = []
    verdicts = []

    # Convert tensor indices back to text (this is just for compatibility)
    # The actual processing will use the tensors directly
    for i in range(len(x)):
        # Create dummy text for compatibility - the actual data will be tensors
        statements.append("dummy_text")
        verdicts.append("real" if y[i].item() == 0 else "fake")

    split_dataset = VeritasDataset(None, statements, verdicts)
    # Override the vocabulary with the original one
    split_dataset.word2idx = word2idx
    split_dataset.idx2word = idx2word

    # Override the __getitem__ method to return the actual tensors
    original_getitem = split_dataset.__getitem__

    def getitem_with_tensors(idx):
        if isinstance(idx, slice):
            return x[idx], y[idx]
        return x[idx], y[idx]

    split_dataset.__getitem__ = getitem_with_tensors

    data_loader = DataLoader(
        split_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader


def run():
    # Uncomment to reprocess all datasets
    # process_data()

    # Create the main dataset to get vocabulary
    main_dataset = VeritasDataset(_DATASET_NAME)

    x_train, x_val, y_train, y_val = split_dataset()
    train_dataloader = create_loader(
        x_train, y_train, word2idx=main_dataset.word2idx, idx2word=main_dataset.idx2word
    )
    val_dataloader = create_loader(
        x_val, y_val, word2idx=main_dataset.word2idx, idx2word=main_dataset.idx2word
    )
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    train_loader, val_loader = run()
    print("Dataset processing complete. DataLoader created.")
