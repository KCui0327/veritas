"""
Description: This script creates the Veritas dataset by running all processor scripts in the specified directory,
combining their outputs, and saving the final dataset to a CSV file

Author: Kenny Cui
Date: July 6, 2025
"""

import hashlib
import pandas as pd
import glob
import runpy
from dataset import VeritasDataset
import os

_DATASET_PATH = "./output"
_DATASET_NAME = "veritas_dataset.csv"

_PROCESSOR_PATH = "../Processor"

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
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame

# Remove duplicates based on the 'statement' column through unique hashing
df["id"] = df["statement"].apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())
df.drop_duplicates(subset=["id"], inplace=True, keep="first")

df.to_csv("veritas_dataset.csv", index=False, encoding="utf-8")

# Create a VeritasDataset instance
dataset = VeritasDataset(_DATASET_NAME)
