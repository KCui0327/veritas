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

_DATASET_PATH = "./output"
_DATASET_NAME = os.path.join(os.path.dirname(__file__), "veritas_dataset.csv")
_PROCESSOR_PATH = "processor"
_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


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

    df.to_csv("veritas_dataset.csv", index=False, encoding="utf-8")


def split_dataset(validation_size=0.2, transform=None):
    # Create a VeritasDataset instance
    dataset = VeritasDataset(_DATASET_NAME)

    statements, verdicts = [], []

    for i in range(len(dataset)):
        statements.append(dataset[i][0])
        verdicts.append(dataset[i][1])

    # Stratified split of the dataset
    x_train, x_val, y_train, y_val = train_test_split(
        statements,
        verdicts,
        test_size=validation_size,
        random_state=1,  # Seed
        stratify=verdicts,  # Enable Stratified split
    )

    return x_train, x_val, y_train, y_val


def create_loader(x, y, batch_size=32, shuffle=False, num_workers=0):
    val_dataset = VeritasDataset(None, x, y)
    data_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader


def run():
    # Uncomment to reprocess all datasets
    # process_data()
    x_train, x_val, y_train, y_val = split_dataset()
    train_dataloader = create_loader(x_train, y_train)
    val_dataloader = create_loader(x_val, y_val)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    train_loader, val_loader = run()
    print("Dataset processing complete. DataLoader created.")
