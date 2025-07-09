"""
Description: This script processes the ISOT dataset by reading all CSV files in the specified directory,
combining them into a single DataFrame, shuffling the rows, and renaming columns.

The final DataFrame is saved to a new CSV file.

Author: Kenny Cui
Date: July 6, 2025
Inspiration from this follow code from Stack Overflow: https://bit.ly/46stqzy
"""

import pandas as pd
import glob
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
_DATA_PATH = os.path.join(PROJECT_DIR, "data")
_OUTPUT_PATH = os.path.join(Path(__file__).resolve().parents[1], "output")

dataset_files = glob.glob(f"{_DATA_PATH}/ISOT/*.csv")
dataset = []
for file in dataset_files:
    print(f"Processing {file}")
    df = pd.read_csv(file, header=0)
    if "True" in file:
        df["verdict"] = "true"
    else:
        df["verdict"] = "false"
    dataset.append(df)

df = pd.concat(dataset, ignore_index=True)
df = df.sample(frac=1, random_state=21).reset_index(drop=True)  # Shuffle the DataFrame
df.rename(columns={"title": "statement", "text": "content"}, inplace=True)
df.loc[df["statement"].notnull(), "statement"] = df[
    "statement"
].str.strip()  # Strip useless characters
df["verdict"] = df["verdict"].str.lower()  # Normalize verdicts

df.to_csv(f"{_OUTPUT_PATH}/ISOT.csv", index=False)
print("ISOT dataset processed and saved to ISOT.csv")
