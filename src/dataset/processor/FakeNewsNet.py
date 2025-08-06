"""
Description: This script processes the FakeNewsNet dataset by reading multiple CSV files,
assigning verdicts based on the filename, and saving the combined dataset to a new CSV file.

The final DataFrame is saved to a new CSV file.

Author: Kenny Cui
Date: July 6, 2025
"""

import pandas as pd
import glob
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
_DATA_PATH = os.path.join(PROJECT_DIR, "data")
_OUTPUT_PATH = os.path.join(Path(__file__).resolve().parents[1], "output")

dataset_files = glob.glob(f"{_DATA_PATH}/FakeNewsNet/*.csv")
dataset = []
for file in dataset_files:
    print(f"Processing {file}")
    df = pd.read_csv(file, header=0)
    if "real" in file:
        df["verdict"] = "true"
    else:
        df["verdict"] = "false"
    df = df.drop(columns=["id"])
    dataset.append(df)

df = pd.concat(dataset, ignore_index=True)
df = df.sample(frac=1, random_state=21).reset_index(drop=True)  # Shuffle the DataFrame
df.rename(
    columns={"title": "statement", "text": "content", "publish_date": "date"},
    inplace=True,
)

# Removing unnecessary columns
del df["movies"]
del df["images"]
del df["top_img"]
del df["canonical_link"]
del df["authors"]
del df["source"]
del df["url"]
del df["meta_data"]

df.loc[df["statement"].notnull(), "statement"] = df[
    "statement"
].str.strip()  # Strip useless characters
df["verdict"] = df["verdict"].str.lower()  # Normalize verdicts

df.to_csv(f"{_OUTPUT_PATH}/FakeNewsNet.csv", index=False)
print("FakeNewsNet dataset processed and saved to FakeNewsNet.csv")
